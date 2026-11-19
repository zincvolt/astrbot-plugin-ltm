"""Long-term memory for QQ chats.

Extracts durable facts about each sender after a reply, stores them with
embeddings, and injects the most relevant memories back into the system
prompt before the next LLM request.

The plugin never touches the LLM deployment itself: it only calls the
OpenAI-compatible endpoints already running on this machine.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

from astrbot.api import logger, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest

DATA_DIR = Path(os.environ.get("QQNTBOT_DATA", r"D:\QQNTBOT\data"))
DB_PATH = DATA_DIR / "memory.db"
BOT_DB = Path(r"D:\QQNTBOT\apps\AstrBot\data\data_v4.db")

SCAN_INTERVAL = 1800  # 每 30 分钟扫一次群历史
SCAN_BUDGET = 1  # 每轮最多为 1 个人抽取记忆

LLM_BASE = "http://127.0.0.1:8080/v1"
LLM_KEY = "sk-llm-7d41c9e2"
LLM_MODEL = "Grey-9B-Heretic"

EMB_BASE = "http://127.0.0.1:8097/v1"
EMB_MODEL = "bge-small-zh-v1.5"

RECALL_TOP_K = 8
RECALL_MIN_SCORE = 0.32
MAX_INJECTED = 4
INJECT_MAX_CHARS = 600      # 记忆块总长度上限（约 400 token，64K 上下文的 0.6%）
INJECT_LINE_CHARS = 60      # 单条记忆截断长度
EXTRACT_COOLDOWN = 45
MAX_FACT_CHARS = 120
MAX_FACTS_PER_USER = 300
DEDUP_SCORE = 0.92

PROMPT_FILE = Path(r"D:\QQNTBOT\data\ltm_prompt.txt")

EXTRACT_PROMPT = """从下面这段对话里提取关于「用户」的、可长期记住的稳定事实。

硬性要求：
- 必须是用户自己原话里明确说过的，不能是推测、补全或从上下文猜的
- 只收这五类：身份/称呼、长期偏好、固定习惯、和谁的关系、长期在做的事
- 不收：一次性闲聊、临时的情绪和状态、别人的事、任何你不确定的内容
- 每条事实不超过 30 字，且必须附上证据 evidence
- evidence 必须是用户原话里一字不差出现的片段（4~20 字），不能改写

只输出 JSON 数组，格式：
[{"fact": "用户的陈述", "evidence": "用户原话片段"}]
没有可记的就输出 []，不要任何解释。

对话：
用户：{user}
对方：{bot}
"""


def load_prompt() -> str:
    """Return the extraction prompt, preferring the editable file.

    Returns:
        Prompt template containing ``{user}`` and ``{bot}`` placeholders.
    """
    try:
        text = PROMPT_FILE.read_text(encoding="utf-8")
        if "{user}" in text and "{bot}" in text:
            return text
        logger.warning("[ltm] ltm_prompt.txt 缺少占位符，回落到内置提示词")
    except OSError:
        pass
    return EXTRACT_PROMPT

def _cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity of two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class Main(star.Star):
    """AstrBot plugin that gives the QQ bot long-term memory."""

    def __init__(self, context: star.Context) -> None:
        self.context = context
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._last_extract: dict[str, float] = {}
        self._init_db()
        self._scan_task: asyncio.Task | None = None
        self._last_hist_id = self._max_history_id()
        logger.info(
            f"[ltm] memory db ready at {DB_PATH} (history cursor={self._last_hist_id})"
        )

    @staticmethod
    def _max_history_id() -> int:
        """Return the newest group-history row id, or 0 when unavailable."""
        if not BOT_DB.exists():
            return 0
        con = sqlite3.connect(f"file:{BOT_DB}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute("SELECT MAX(id) FROM platform_message_history").fetchone()
        except sqlite3.Error:
            return 0
        finally:
            con.close()
        return int(row[0] or 0) if row else 0

    # ---------- storage ----------

    def _conn(self) -> sqlite3.Connection:
        """Open a short-lived SQLite connection."""
        con = sqlite3.connect(DB_PATH, timeout=10)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self) -> None:
        """Create the facts table when missing."""
        con = self._conn()
        try:
            con.execute(
                """CREATE TABLE IF NOT EXISTS facts (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       uid TEXT NOT NULL,
                       text TEXT NOT NULL,
                       emb TEXT,
                       created REAL NOT NULL,
                       used REAL,
                       hits INTEGER DEFAULT 0
                   )"""
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_facts_uid ON facts(uid)")
            con.commit()
        finally:
            con.close()

    # ---------- llm / embedding helpers ----------

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embed texts with the local NPU service; returns None when unavailable."""
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    f"{EMB_BASE}/embeddings",
                    json={"model": EMB_MODEL, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                return [item["embedding"] for item in data]
        except Exception as exc:  # noqa: BLE001 - degrade to keyword-free recall
            logger.warning(f"[ltm] embedding unavailable: {exc}")
            return None

    async def _chat(self, prompt: str, max_tokens: int = 220) -> str:
        """Run one short completion against the local model."""
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                f"{LLM_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_KEY}"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""

    # ---------- recall ----------

    def _recent(self, uid: str, limit: int) -> list[str]:
        """Return the most recently stored facts for a user."""
        con = self._conn()
        try:
            rows = con.execute(
                "SELECT text FROM facts WHERE uid = ? ORDER BY created DESC LIMIT ?",
                (uid, limit),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            con.close()

    async def _recall(self, uid: str, query: str) -> list[str]:
        """Pick memories relevant to the current message."""
        con = self._conn()
        try:
            rows = con.execute(
                "SELECT id, text, emb, created FROM facts WHERE uid = ? AND created >= ? "
            "ORDER BY created DESC LIMIT 400",
                (uid, time.time() - 30 * 86400),
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return []

        vectors = await self._embed([query])
        if not vectors:
            # No embedding service: fall back to the most recent memories.
            return [r[1] for r in rows[:MAX_INJECTED]]

        qv = vectors[0]
        scored: list[tuple[float, str]] = []
        for fact_id, text, emb, _created in rows:
            try:
                fv = json.loads(emb) if emb else None
            except json.JSONDecodeError:
                fv = None
            score = _cosine(qv, fv) if fv else 0.0
            if score >= RECALL_MIN_SCORE:
                scored.append((score, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [t for _, t in scored[:RECALL_TOP_K]]

        # Always keep a few recent memories so the bot does not forget context.
        for text in self._recent(uid, 4):
            if text not in picked:
                picked.append(text)

        return picked[:MAX_INJECTED]

    # ---------- extraction ----------

    async def _extract(self, uid: str, user_text: str, bot_text: str) -> None:
        """Ask the model for durable facts and persist the new ones."""
        try:
            raw = await self._chat(
                load_prompt().replace("{user}", user_text[:800]).replace(
                "{bot}", bot_text[:800]
            )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[ltm] extract call failed: {exc}")
            return

        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return
        if not isinstance(items, list):
            return

        flat_user = re.sub(r"\s+", "", user_text)
        banned = ("可能", "也许", "好像", "估计", "应该", "大概", "似乎")
        candidates = []
        for item in items:
            if isinstance(item, dict):
                text = str(item.get("fact", "")).strip()
                evidence = re.sub(r"\s+", "", str(item.get("evidence", "")))
            else:
                continue
            # 证据必须能在用户原话里找到，否则视为编造
            if not evidence or evidence not in flat_user:
                logger.debug(f"[ltm] drop unverified fact: {text[:30]}")
                continue
            if any(word in text for word in banned):
                continue
            if not 6 <= len(text) <= MAX_FACT_CHARS:
                continue
            candidates.append(text)
            if len(candidates) >= 2:
                break
        if not candidates:
            return

        vectors = await self._embed(candidates) or []

        con = self._conn()
        added = 0
        try:
            existing = con.execute(
                "SELECT text, emb FROM facts WHERE uid = ? ORDER BY created DESC LIMIT 300",
                (uid,),
            ).fetchall()
            existing_vecs = []
            for _t, emb in existing:
                try:
                    existing_vecs.append(json.loads(emb) if emb else [])
                except json.JSONDecodeError:
                    existing_vecs.append([])

            now = time.time()
            for idx, text in enumerate(candidates):
                vec = vectors[idx] if idx < len(vectors) else []
                if vec and any(
                    _cosine(vec, other) >= DEDUP_SCORE for other in existing_vecs if other
                ):
                    continue
                con.execute(
                    "INSERT INTO facts (uid, text, emb, created) VALUES (?, ?, ?, ?)",
                    (uid, text, json.dumps(vec) if vec else None, now),
                )
                if vec:
                    existing_vecs.append(vec)
                added += 1

            con.execute(
                """DELETE FROM facts WHERE uid = ? AND id NOT IN (
                       SELECT id FROM facts WHERE uid = ? ORDER BY created DESC LIMIT ?)""",
                (uid, uid, MAX_FACTS_PER_USER),
            )
            con.commit()
            if added:
                logger.info(f"[ltm] stored {added} new fact(s) for {uid}")
        finally:
            con.close()

    # ---------- hooks ----------

    @filter.on_llm_request()
    async def inject_memory(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Append relevant memories to the system prompt."""
        uid = event.get_sender_id() or event.get_session_id()
        if not uid:
            return
        query = (event.message_str or "").strip()
        memories = await self._recall(uid, query) if query else self._recent(uid, 6)
        if not memories:
            return
        # 每条截断 + 总量封顶，避免记忆挤占对话上下文
        lines = []
        budget = INJECT_MAX_CHARS
        for mem in memories:
            line = mem if len(mem) <= INJECT_LINE_CHARS else mem[:INJECT_LINE_CHARS] + "…"
            if budget - len(line) < 0:
                break
            lines.append("- " + line)
            budget -= len(line) + 2
        if not lines:
            return
        block = "\n".join(lines)
        req.system_prompt = (req.system_prompt or "") + (
            "\n\n[你长期记得的关于对方的事，自然地用，不要逐条复述]\n" + block
        )

    @filter.on_llm_response()
    async def learn_from_turn(
        self, event: AstrMessageEvent, response: LLMResponse
    ) -> None:
        """Schedule fact extraction after a reply."""
        uid = event.get_sender_id() or event.get_session_id()
        user_text = (event.message_str or "").strip()
        bot_text = (response.completion_text or "").strip()
        if not uid or len(user_text) < 6 or not bot_text:
            return

        now = time.time()
        if now - self._last_extract.get(uid, 0.0) < EXTRACT_COOLDOWN:
            return
        self._last_extract[uid] = now

        asyncio.create_task(self._extract(uid, user_text, bot_text))

    # ---------- group-history scanning ----------

    def _new_group_rows(self) -> list[tuple[int, str, str]]:
        """Return new group messages as (row_id, sender_id, text).

        Skips the bot's own output and any non-group chat.
        """
        if not BOT_DB.exists():
            return []
        con = sqlite3.connect(f"file:{BOT_DB}?mode=ro", uri=True, timeout=5)
        try:
            rows = con.execute(
                "SELECT id, sender_id, user_id, content FROM platform_message_history "
                "WHERE id > ? ORDER BY id ASC LIMIT 300",
                (self._last_hist_id,),
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            con.close()

        out: list[tuple[int, str, str]] = []
        for rid, sender, umo, content in rows:
            if "GroupMessage" not in str(umo):
                continue
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
            if data.get("type") == "bot":
                continue
            parts = [
                str(part.get("text") or "")
                for part in data.get("message", [])
                if part.get("type") in ("text", "plain")
            ]
            text = " ".join(p for p in parts if p).strip()
            if text:
                out.append((int(rid), str(sender), text))
        return out

    @filter.on_astrbot_loaded()
    async def start_scanner(self) -> None:
        """Start the background group-history scanner."""
        if self._scan_task is None:
            self._scan_task = asyncio.create_task(self._scan_loop())
            logger.info("[ltm] group history scanner started")

    async def _scan_loop(self) -> None:
        """Periodically learn facts about everyone who talks in the group."""
        while True:
            await asyncio.sleep(SCAN_INTERVAL)
            try:
                await self._scan_once()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                logger.warning(f"[ltm] scan failed: {exc}")

    def _is_third_party(self) -> bool:
        """Return True when the bot runs on the third-party API.

        Local big models must not be used in that mode, so memory extraction is
        skipped entirely.

        Returns:
            Whether the bot is currently on the third-party API.
        """
        import json
        from pathlib import Path as _Path

        try:
            cfg = json.loads(
                _Path(r"D:\QQNTBOT\apps\AstrBot\data\cmd_config.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            pid = str(
                cfg.get("agent_runner", {})
                .get("config", {})
                .get("model", {})
                .get("provider_id", "")
            )
            if pid:
                return "deepseek" in pid
        except Exception:  # noqa: BLE001
            pass
        state = _Path(r"D:\QQNTBOT\data\api_mode.json")
        if not state.exists():
            return False
        try:
            return json.loads(state.read_text(encoding="utf-8")).get("mode") == "third_party"
        except Exception:  # noqa: BLE001
            return False

    async def _scan_once(self) -> None:
        """Group new messages per sender and extract facts for a few of them."""
        if self._is_third_party():
            logger.info("[ltm] third-party mode: memory scan skipped")
            return
        rows = self._new_group_rows()
        if not rows:
            return
        self._last_hist_id = rows[-1][0]

        buckets: dict[str, list[str]] = {}
        for _rid, uid, text in rows:
            buckets.setdefault(uid, []).append(text)

        budget = SCAN_BUDGET
        for uid, texts in buckets.items():
            if budget <= 0:
                break
            joined = " / ".join(texts)[-400:]
            if len(joined) < 12:
                continue
            budget -= 1
            await self._extract(uid, joined, "（群里聊天，没有直接对话）")
        logger.info(f"[ltm] scanned {len(rows)} group msgs, {len(buckets)} senders")

    # ---------- commands ----------

    @filter.command("记忆")
    async def show_memory(self, event: AstrMessageEvent) -> None:
        """List what the bot remembers about the sender."""
        uid = event.get_sender_id() or event.get_session_id()
        con = self._conn()
        try:
            rows = con.execute(
                "SELECT text FROM facts WHERE uid = ? ORDER BY created DESC LIMIT 20", (uid,)
            ).fetchall()
        finally:
            con.close()
        if not rows:
            yield event.plain_result("我还没记住关于你的事。")
            return
        body = "\n".join(f"{i}. {r[0]}" for i, r in enumerate(rows, 1))
        yield event.plain_result(f"关于 {uid} 的记忆（共 {len(rows)} 条）：\n{body}")

    @filter.command("忘记")
    async def forget(self, event: AstrMessageEvent, keyword: str = "") -> None:
        """Delete memories matching a keyword, or all of them when empty."""
        uid = event.get_sender_id() or event.get_session_id()
        con = self._conn()
        try:
            if keyword:
                cur = con.execute(
                    "DELETE FROM facts WHERE uid = ? AND text LIKE ?",
                    (uid, f"%{keyword}%"),
                )
            else:
                cur = con.execute("DELETE FROM facts WHERE uid = ?", (uid,))
            con.commit()
            deleted = cur.rowcount
        finally:
            con.close()
        yield event.plain_result(f"忘掉了 {deleted} 条。")
