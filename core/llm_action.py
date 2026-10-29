import copy
import random
import re
from datetime import datetime, timedelta
from typing import Any

from astrbot.api import logger
from astrbot.core.provider.provider import Provider

from .config import PluginConfig
from .model import Comment, Post

SCENE_POST = "post"
SCENE_COMMENT = "comment"
SCENE_REPLY = "reply"

SCENE_LABELS = {
    SCENE_POST: "写说说",
    SCENE_COMMENT: "评论说说",
    SCENE_REPLY: "回复评论",
}

TIME_PLACEHOLDER = "{current_time}"
SCENE_PLACEHOLDER = "{scene}"


class LLMAction:
    def __init__(self, config: PluginConfig):
        self.cfg = config
        self.context = config.context

    @staticmethod
    def _join_prompt_parts(*parts: str) -> str:
        return "\n\n".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _normalize_ts(ts: int) -> int:
        """把毫秒级时间戳归一化为秒级，避免 fromtimestamp 溢出"""
        if ts > 10_000_000_000:
            return ts // 1000
        return ts

    @staticmethod
    def _msg_ts(msg: dict[str, Any]) -> int:
        """取出一条 OneBot 消息的时间戳（统一为秒）"""
        for key in ("time", "timestamp", "send_time"):
            value = msg.get(key)
            try:
                if value is not None:
                    return LLMAction._normalize_ts(int(value))
            except (TypeError, ValueError, OverflowError):
                continue
        return 0

    def _is_msg_in_days(self, msg: dict[str, Any], since: datetime) -> bool:
        ts = self._msg_ts(msg)
        if ts <= 0:
            return True
        try:
            msg_dt = datetime.fromtimestamp(ts, self.cfg.timezone)
        except (ValueError, OSError, OverflowError):
            return True
        return msg_dt >= since

    @staticmethod
    def _build_context(
        round_messages: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """
        把所有回合里的纯文本消息打包成 openai-style 的 user 上下文。
        只有存在非空文本片段的消息才会保留，避免把只含图片/表情的消息当成有效内容。
        """
        contexts: list[dict[str, str]] = []
        for msg in round_messages:
            text_segments = [
                str(seg.get("data", {}).get("text") or "")
                for seg in msg.get("message") or []
                if seg.get("type") == "text"
            ]
            text = "".join(text_segments).strip()
            # 仅当真正说了话才保留
            if not text:
                continue
            nickname = (msg.get("sender") or {}).get("nickname") or "未知用户"
            contexts.append({"role": "user", "content": f"{nickname}: {text}"})
        return contexts

    async def _get_msg_contexts(
        self, group_id: str, days: int | None = None
    ) -> list[dict]:
        """按日期回溯获取群聊历史消息，只保留最近 days 天内的内容"""
        message_seq = 0
        days = max(1, days if days is not None else self.cfg.source.post_days)
        # 以本地零点为锚点：days=1 表示今天，days=2 表示今天+昨天
        now = self.cfg.now()
        start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        since = start_of_today - timedelta(days=days - 1)
        contexts: list[dict] = []
        if not self.cfg.client:
            raise RuntimeError("客户端未初始化")
        while len(contexts) < self.cfg.source.post_max_msg:
            payloads = {
                "group_id": group_id,
                "message_seq": message_seq,
                "count": 200,
                "reverseOrder": True,
            }
            result: dict = await self.cfg.client.api.call_action(
                "get_group_msg_history", **payloads
            )
            round_messages = result.get("messages") or []
            if not round_messages:
                break

            # 本批消息已回溯到指定日期之前，取完这一批里的有效消息即可结束
            in_range = [
                msg for msg in round_messages if self._is_msg_in_days(msg, since)
            ]
            contexts.extend(self._build_context(in_range))

            if len(in_range) < len(round_messages):
                break
            message_seq = round_messages[0]["message_id"]

        # 分页是从新到旧拉取，合起来整体是倒序；反转后按时间正序，
        # 再保留末尾（即最近）的 post_max_msg 条，确保截断丢掉的是最旧的消息
        contexts.reverse()
        return contexts[-self.cfg.source.post_max_msg :]

    async def _resolve_msg_contexts(
        self, group_id: str = ""
    ) -> tuple[str, list[dict], list[str]]:
        """
        确定用于写说说的聊天记录。

        - 指定 group_id 时只取该会话；
        - 否则随机抽取若干候选群聊，逐个尝试；某个会话在指定日期内没有聊天记录，
          就切换到下一个会话，直到取到非空聊天记录为止。
        返回 (实际使用的会话ID, 聊天记录, 尝试过的会话列表)
        """
        if not self.cfg.client:
            raise RuntimeError("客户端未初始化")

        tried: list[str] = []
        if group_id:
            tried.append(str(group_id))
            return str(group_id), await self._get_msg_contexts(str(group_id)), tried

        group_list = await self.cfg.client.get_group_list()
        group_ids = [
            str(group["group_id"])
            for group in group_list
            if str(group["group_id"]) not in self.cfg.source.ignore_groups
        ]
        if not group_ids:
            return "", [], []

        pool_size = max(1, min(int(self.cfg.source.group_session_pool), len(group_ids)))
        candidates = random.sample(group_ids, pool_size)

        for candidate in candidates:
            tried.append(candidate)
            try:
                contexts = await self._get_msg_contexts(candidate)
            except Exception as e:
                logger.warning(f"获取群聊 {candidate} 的聊天记录失败，切换下一个会话: {e}")
                continue
            if contexts:
                if len(tried) > 1:
                    logger.info(
                        f"前 {len(tried) - 1} 个会话在指定日期内无聊天记录，改用会话 {candidate}"
                    )
                return candidate, contexts, tried

        logger.warning(f"候选会话在指定日期内均无聊天记录：{tried}")
        return "", [], tried

    @staticmethod
    def _get_event_umo(event: Any | None) -> str | None:
        umo = getattr(event, "unified_msg_origin", None)
        return str(umo) if umo else None

    def _get_event_platform_name(self, event: Any | None) -> str:
        getter = getattr(event, "get_platform_name", None)
        if callable(getter):
            try:
                platform_name = getter()
            except Exception:
                platform_name = None
            if platform_name:
                return str(platform_name)

        umo = self._get_event_umo(event)
        if umo and ":" in umo:
            return umo.split(":", 1)[0]
        return ""

    def _get_provider_settings(self, event: Any | None) -> dict[str, Any]:
        umo = self._get_event_umo(event)
        cfg = self.context.get_config(umo) if umo else self.context.get_config()
        provider_settings = cfg.get("provider_settings", {})
        return provider_settings if isinstance(provider_settings, dict) else {}

    def _get_provider(
        self, provider_id: str, event: Any | None = None
    ) -> Provider | None:
        provider = self.context.get_provider_by_id(provider_id) if provider_id else None
        if isinstance(provider, Provider):
            return provider

        try:
            umo = self._get_event_umo(event)
            provider = (
                self.context.get_using_provider(umo)
                if umo
                else self.context.get_using_provider()
            )
        except Exception as e:
            logger.error(f"获取当前会话的 LLM 提供商失败: {e}")
            return None

        return provider if isinstance(provider, Provider) else None

    async def _get_persona_context(
        self, event: Any | None
    ) -> tuple[str, list[dict[str, Any]]]:
        if not event:
            return "", []

        umo = self._get_event_umo(event)
        if not umo:
            return "", []

        try:
            conversation_persona_id = None
            cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if cid:
                conversation = await self.context.conversation_manager.get_conversation(
                    umo, cid
                )
                if conversation:
                    conversation_persona_id = conversation.persona_id

            (
                persona_id,
                persona,
                _,
                _,
            ) = await self.context.persona_manager.resolve_selected_persona(
                umo=umo,
                conversation_persona_id=conversation_persona_id,
                platform_name=self._get_event_platform_name(event),
                provider_settings=self._get_provider_settings(event),
            )

            if not persona and persona_id:
                persona = self.context.persona_manager.get_persona_v3_by_id(persona_id)

            if not persona:
                return "", []

            persona_prompt = str(persona.get("prompt") or "").strip()
            begin_dialogs = copy.deepcopy(persona.get("_begin_dialogs_processed") or [])
            return persona_prompt, begin_dialogs
        except Exception as e:
            logger.warning(f"解析当前会话人格失败，将仅使用插件提示词: {e}")
            return "", []

    def _render_prompt(self, text: str, scene: str) -> str:
        """填充提示词占位符：{current_time} / {scene}"""
        if not text:
            return ""
        return (
            text.replace(TIME_PLACEHOLDER, self.cfg.now_str())
            .replace(SCENE_PLACEHOLDER, SCENE_LABELS.get(scene, scene))
            .strip()
        )

    def _build_task_prompt(
        self, scene: str, scene_prompt: str, extra: str = ""
    ) -> tuple[str, bool]:
        """
        统合提示词：[全局主提示词] + [当前场景独立提示词] + [可选场景附加要求]
        每个场景只替换自己的独立提示词，主提示词始终共享。
        返回 (task_prompt, has_time_placeholder)
        """
        has_time = any(
            TIME_PLACEHOLDER in (part or "")
            for part in (self.cfg.llm.main_prompt, scene_prompt, extra)
        )
        task_prompt = self._join_prompt_parts(
            self._render_prompt(self.cfg.llm.main_prompt, scene),
            self._render_prompt(scene_prompt, scene),
            self._render_prompt(extra, scene),
        )
        return task_prompt, has_time

    async def _build_request_context(
        self,
        *,
        event: Any | None,
        task_prompt: str,
        prompt_has_time: bool = False,
        contexts: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        persona_prompt, persona_contexts = await self._get_persona_context(event)
        parts = [
            "# Persona Instructions\n\n" + persona_prompt if persona_prompt else "",
            task_prompt,
        ]
        # 提示词里已经有时间占位符或显式包含当前时间段落时不再追加，避免当前时间被重复声明
        has_time = (
            prompt_has_time
            or "# 当前时间" in task_prompt
            or (persona_prompt and TIME_PLACEHOLDER in persona_prompt)
        )
        if not has_time:
            parts.append(f"# 当前时间\n{self.cfg.now_str()}")
        system_prompt = self._join_prompt_parts(*parts)
        merged_contexts = [*persona_contexts]
        if contexts:
            merged_contexts.extend(contexts)
        return system_prompt, merged_contexts

    @staticmethod
    def extract_content(raw: str) -> str:
        start_marker = '"""'
        end_marker = '"""'
        start = raw.find(start_marker) + len(start_marker)
        end = raw.find(end_marker, start)
        if start != -1 and end != -1:
            return raw[start:end].strip()
        return ""

    async def generate_post(
        self,
        group_id: str = "",
        topic: str | None = None,
        *,
        event: Any | None = None,
    ) -> str | None:
        """生成帖子"""
        provider = self._get_provider(self.cfg.llm.post_provider_id, event)
        if not isinstance(provider, Provider):
            raise RuntimeError("未配置用于文本生成任务的 LLM 提供商")

        if not self.cfg.client:
            raise RuntimeError("客户端未初始化")

        group_id, contexts, _ = await self._resolve_msg_contexts(group_id)
        if not contexts:
            logger.warning("选定的会话没有可用于写说说的聊天记录")
            return None
        # TODO: 更多模式

        task_prompt, has_time = self._build_task_prompt(
            SCENE_POST,
            self.cfg.llm.post_prompt,
            f"# 写作主题：{topic or '从聊天内容中选一个主题'}",
        )
        task_prompt = self._join_prompt_parts(
            task_prompt,
            "# 输出格式要求：\n"
            '- 使用三对双引号（"""）将正文内容包裹起来。\n'
            "- 只输出最终可发布的说说正文，不要附带解释、标题或额外说明。",
        )
        system_prompt, contexts = await self._build_request_context(
            event=event,
            task_prompt=task_prompt,
            prompt_has_time=has_time,
            contexts=contexts,
        )

        logger.debug(f"{system_prompt}\n\n{contexts}")

        try:
            llm_response = await provider.text_chat(
                system_prompt=system_prompt,
                contexts=contexts,
            )
            diary = self.extract_content(llm_response.completion_text)
            if not diary:
                raise ValueError("LLM 生成的日记为空")
            logger.info(f"LLM 生成的日记：{diary}")
            return diary

        except Exception as e:
            raise ValueError(f"LLM 调用失败：{e}")

    async def generate_comment(
        self, post: Post, *, event: Any | None = None
    ) -> str | None:
        """根据帖子内容生成评论"""
        provider = self._get_provider(self.cfg.llm.comment_provider_id, event)
        if not isinstance(provider, Provider):
            logger.error("未配置用于文本生成任务的 LLM 提供商")
            return None
        try:
            content = post.text
            if post.rt_con:  # 转发文本
                content += f"\n[转发]\n{post.rt_con}"

            prompt = f"\n[帖子内容]：\n{content}"
            task_prompt, has_time = self._build_task_prompt(
                SCENE_COMMENT, self.cfg.llm.comment_prompt
            )
            system_prompt, contexts = await self._build_request_context(
                event=event,
                task_prompt=self._join_prompt_parts(
                    task_prompt,
                    "# 输出要求：\n- 只输出最终评论内容，不要解释，不要分点，不要添加额外前缀。",
                ),
                prompt_has_time=has_time,
            )

            logger.debug(prompt)
            # 先尝试带图片调用，若 LLM 不支持 vision 则降级为纯文本
            try:
                llm_response = await provider.text_chat(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    contexts=contexts,
                    image_urls=post.images if post.images else None,
                )
            except Exception as img_err:
                err_msg = str(img_err).lower()
                if "image_url" in err_msg or "image" in err_msg:
                    logger.warning(
                        f"当前 LLM 不支持 vision，降级为纯文本评论: {img_err}"
                    )
                    llm_response = await provider.text_chat(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        contexts=contexts,
                    )
                else:
                    raise
            comment = re.sub(r"[\s\u3000]+", "", llm_response.completion_text).rstrip(
                "。"
            )
            logger.info(f"LLM 生成的评论：{comment}")
            return comment

        except Exception as e:
            raise ValueError(f"LLM 调用失败：{e}")

    async def generate_reply(
        self,
        post: Post,
        comment: Comment,
        *,
        event: Any | None = None,
    ) -> str | None:
        """根据评论内容生成回复"""
        provider = self._get_provider(self.cfg.llm.reply_provider_id, event)
        if not isinstance(provider, Provider):
            logger.error("未配置用于文本生成任务的 LLM 提供商")
            return None
        try:
            content = post.text
            if post.rt_con:  # 转发文本
                content += f"\n[转发]\n{post.rt_con}"

            prompt = f"\n## 帖子内容\n{content}"
            prompt += f"\n## 要回复的评论\n{comment.nickname}：{comment.content}"
            task_prompt, has_time = self._build_task_prompt(
                SCENE_REPLY, self.cfg.llm.reply_prompt
            )
            system_prompt, contexts = await self._build_request_context(
                event=event,
                task_prompt=self._join_prompt_parts(
                    task_prompt,
                    "# 输出要求：\n- 只输出最终回复内容，不要解释，不要分点，不要添加额外前缀。",
                ),
                prompt_has_time=has_time,
            )
            logger.debug(prompt)
            llm_response = await provider.text_chat(
                system_prompt=system_prompt,
                prompt=prompt,
                contexts=contexts,
            )
            reply = re.sub(r"[\s\u3000]+", "", llm_response.completion_text).rstrip(
                "。"
            )
            logger.info(f"LLM 生成的回复：{reply}")
            return reply

        except Exception as e:
            raise ValueError(f"LLM 调用失败：{e}")
