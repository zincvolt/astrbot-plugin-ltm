"""Long-term memory for QQ chats, tuned for a capable LLM on a budget.

Design goals (v4 rewrite, 2026-11-20):

1. Few LLM calls. The old version scanned every 90s and could fire dozens of
   DeepSeek calls per hour, most of which came back empty. Now the scanner
   only accumulates messages; a person is only sent to the model when a real
   batch has piled up (>= MIN_BATCH usable lines) or a lot of time passed.
   Group replies never trigger extraction directly.
2. Slot-keyed storage. Identity-type slots (name/age/location/identity/pet/
   relation) keep exactly one live row per slot; a new value supersedes the
   old one. This kills the "养了只傻猫 / 骚猫 / 乐子猫" duplicate swarm that
   wording drift used to slip past embedding dedupe.
3. The model sees what is already known, so it stops re-extracting facts.
4. JSON output. A capable model returns clean JSON; no more verbatim
   grounding of tiny snippets and no rule-based miner fallback.
"""

from __future__ import annotations

import asyncio
import difflib
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

def _detect_root() -> Path:
    """Locate the project root so the plugin works outside this machine.

    Returns:
        Directory containing ``data/`` and ``apps/AstrBot/``.
    """
    env = os.environ.get("QQNTBOT_ROOT", "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").is_dir() and (parent / "apps" / "AstrBot").is_dir():
            return parent
    # 标准布局：<root>/apps/AstrBot/data/plugins/<plugin>/main.py
    return here.parents[4] if len(here.parents) > 4 else here.parent


# 指令自声明：/权限 列表 按插件分组展示用（插件开发规范要求，见 docs/PLUGIN-DEV.md）
DECLARED_COMMANDS: dict[str, tuple[str, str]] = {
    "/记忆 <内容>": ("记一条长期记忆", "公开"),
    "/忘记 <关键词>": ("删除匹配的长期记忆", "公开"),
}

ROOT = _detect_root()
DATA_DIR = Path(os.environ.get("QQNTBOT_DATA", str(ROOT / "data")))
DB_PATH = DATA_DIR / "memory.db"
CONFIG_FILE = DATA_DIR / "ltm.json"
BOT_DB = Path(
    os.environ.get(
        "LTM_BOT_DB", str(ROOT / "apps" / "AstrBot" / "data" / "data_v4.db")
    )
)
CMD_CONFIG = Path(
    os.environ.get(
        "LTM_CMD_CONFIG", str(ROOT / "apps" / "AstrBot" / "data" / "cmd_config.json")
    )
)

DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "scan_interval": 300,      # 多久醒一次攒群消息（秒）；醒来本身不调用模型
    "scan_budget": 3,          # 每轮最多几个人被送模型
    "user_cooldown": 1800,     # 同一个人两次提取的最小间隔（秒）
    "extract_cooldown": 900,   # 私聊回复触发的最小间隔（秒）
    "max_tokens": 300,
    "third_party_scan": True,  # 第三方模式下是否也扫群历史（会产生 API 费用）
}

# 本地模式的兜底端点；正常情况由 cmd_config.json 的 provider 配置决定
FALLBACK_LLM = ("http://127.0.0.1:8080/v1", "sk-llm-7d41c9e2", "Grey-9B-Heretic")

EMB_BASE = os.environ.get("LTM_EMB_BASE", "http://127.0.0.1:8097/v1").rstrip("/")
EMB_MODEL = os.environ.get("LTM_EMB_MODEL", "bge-small-zh-v1.5")

RECALL_TOP_K = 8
RECALL_MIN_SCORE = 0.32
MAX_INJECTED = 4
INJECT_MAX_CHARS = 600      # 记忆块总长度上限（约 400 token，64K 上下文的 0.6%）
INJECT_LINE_CHARS = 60      # 单条记忆截断长度
MAX_FACT_CHARS = 120
MAX_FACTS_PER_USER = 300
MULTI_SLOT_CAP = 6          # 多值槽位（喜欢/讨厌/习惯/在做的事）每人每槽最多几条
MAX_CANDIDATES = 8          # 一次提取最多入库几条
EXTRACT_BATCH = 40          # 一次送给模型的消息上限
MIN_BATCH = 8               # 积压满这么多条可用消息才值得一次调用
MAX_WAIT = 1800             # 积压不足时，最多等这么久也照样调用一次（秒）

DEDUP_TEXT = 0.85           # 多值槽位：文本相似判重阈值
DEDUP_EMB = 0.88            # 多值槽位：向量判重阈值

# 单值槽位：一个 uid 每个槽只留一条 live 记忆，新值直接顶掉旧值
SINGLE_SLOTS = {"name", "age", "location", "identity", "pet", "relation"}
# 多值槽位：可以并存多条，靠相似度去重 + FIFO 上限
MULTI_SLOTS = {"like", "dislike", "habit", "activity"}
SLOT_WHITELIST = SINGLE_SLOTS | MULTI_SLOTS

# 这些槽位说的是「最近在干什么」，会过期，注入时必须带时间
TIMED_SLOTS = {"activity", "habit"}

# 模型把值写成这些词等于没记
USELESS_VALUES = {
    "你", "他", "她", "它", "你们", "他们", "她们", "这个", "那个", "这些",
    "那些", "什么", "怎么", "这样", "那样", "一下", "一点", "真的", "反正",
    "哈哈", "呵呵", "东西", "事情", "问题", "感觉", "自己", "对方", "人家",
    "本人", "一个人", "有点", "比较", "非常", "特别", "没", "没有", "不是",
    "不知道", "算了", "随便", "都行", "一样", "null", "none", "无",
}

# relation 的值必须能看出是什么关系，否则「那个群有俩人」也会被当成家人
RELATION_WORDS = (
    "妈", "爸", "爹", "娘", "哥", "姐", "弟", "妹", "对象", "女友", "男友",
    "老婆", "老公", "媳妇", "室友", "舍友", "朋友", "同事", "同学", "师父",
    "徒弟", "儿子", "女儿", "家人", "家里",
)

SLOT_TEMPLATES = {
    "name": "对方叫{value}",
    "age": "对方{value}岁",
    "location": "对方在{value}",
    "identity": "对方是{value}",
    "like": "对方喜欢{value}",
    "dislike": "对方不喜欢{value}",
    "habit": "对方经常{value}",
    "pet": "对方养了{value}",
    "relation": "对方的{value}",
    "activity": "对方{value}",
}

EXTRACT_PROMPT = """你在帮一个 QQ 群聊机器人维护它对某位群友的长期记忆。先给出已经记住的内容，再给出这位群友最近说过的话。

【已记住】
{known}

【最近说的话】
{lines}

任务：从这些话里找出「长期成立、值得记住」的新事实，或者对已有记忆的更新。

判断标准：
- 只记说话人自己的事：身份职业、常住地、宠物、喜欢/讨厌、长期习惯、最近在做的事、人际关系
- 心情、玩笑、接梗、对别人的评论、一次性的事不记；带「今天/昨天/刚才」的临时状态不记
- 「在打排位」「在赶稿」这类进行中的事记为 activity，会过期但值得记
- value 写简短自然的短语（15 字以内），可以浓缩原话，不必逐字抄
- 【已记住】里仍然正确的内容不要重复输出；只有实质变化才输出新值

输出一个 JSON 数组，每项形如 {"slot": "槽位", "value": "事实", "ref": 原话编号}，例如：
[{"slot": "location", "value": "南京", "ref": 3}, {"slot": "pet", "value": "一只叫阿灰的猫", "ref": 4}]
没有值得记的就输出 []。只输出 JSON，不要输出任何解释。"""

PROMPT_FILE = Path(os.environ.get("LTM_PROMPT", str(DATA_DIR / "ltm_prompt.txt")))

# (mtime, settings) / (mtime, (base, key, model)) — 配置文件改动后自动失效
_settings_cache: tuple[float, dict[str, Any]] | None = None
_llm_cache: tuple[float, tuple[str, str, str]] | None = None


def _resolve_llm() -> tuple[str, str, str]:
    """Resolve the chat endpoint of the provider AstrBot is currently using.

    Returns:
        Tuple of (api_base, api_key, model).
    """
    global _llm_cache
    try:
        mtime = CMD_CONFIG.stat().st_mtime
    except OSError:
        return FALLBACK_LLM
    if _llm_cache and _llm_cache[0] == mtime:
        return _llm_cache[1]

    try:
        cfg = json.loads(CMD_CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError):
        return FALLBACK_LLM

    pid = str(
        cfg.get("agent_runner", {}).get("config", {}).get("model", {}).get("provider_id", "")
    )
    provider = next((p for p in cfg.get("provider", []) if p.get("id") == pid), None)
    if not provider:
        logger.warning(f"[ltm] 找不到 provider {pid!r}，用本地兜底端点")
        return FALLBACK_LLM

    source = next(
        (
            s
            for s in cfg.get("provider_sources", [])
            if s.get("id") == provider.get("provider_source_id")
        ),
        None,
    )
    if not source:
        return FALLBACK_LLM

    base = str(source.get("api_base") or "").rstrip("/")
    keys = source.get("key") or []
    key = source.get("api_key") or (keys[0] if keys else "")
    model = str(provider.get("model") or "")
    if not base or not key or not model:
        return FALLBACK_LLM

    resolved = (base, key, model)
    _llm_cache = (mtime, resolved)
    logger.info(f"[ltm] extract via provider={pid} model={model}")
    return resolved


def _looks_mojibake(text: str) -> bool:
    """Detect UTF-8 text that was decoded as Latin-1 and re-saved.

    Args:
        text: Prompt text read from disk.

    Returns:
        True when the text looks like double-encoded Chinese.
    """
    latin = sum(1 for ch in text if "\x80" <= ch <= "\xff")
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return latin > 10 and cjk < latin * 0.2


def _demojibake(text: str) -> str | None:
    """Try to undo a Latin-1/CP1252 mis-decode.

    Args:
        text: Suspect text.

    Returns:
        Recovered text when it decodes back to valid UTF-8 Chinese, else None.
    """
    for enc in ("latin-1", "cp1252"):
        try:
            fixed = text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if sum(1 for ch in fixed if "\u4e00" <= ch <= "\u9fff") > 5:
            return fixed
    return None


def load_settings() -> dict[str, Any]:
    """Return merged settings, re-reading ltm.json only when it changed.

    Returns:
        Settings dict with defaults filled in.
    """
    global _settings_cache
    try:
        mtime = CONFIG_FILE.stat().st_mtime
    except OSError:
        return dict(DEFAULT_SETTINGS)
    if _settings_cache and _settings_cache[0] == mtime:
        return _settings_cache[1]
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("ltm.json 必须是对象")
        merged = {**DEFAULT_SETTINGS, **raw}
        _settings_cache = (mtime, merged)
        return merged
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"[ltm] ltm.json 读取失败，用默认值: {exc}")
        return dict(DEFAULT_SETTINGS)


def load_prompt() -> str:
    """Return the extraction prompt, preferring the editable file.

    The v4 prompt needs both ``{lines}`` (the messages) and ``{known}`` (the
    slot summary). Older files with only ``{lines}`` are still accepted; the
    missing ``{known}`` degrades to "no known facts shown".

    Returns:
        Prompt template.
    """
    try:
        text = PROMPT_FILE.read_text(encoding="utf-8")
    except OSError:
        return EXTRACT_PROMPT

    def ok(tpl: str) -> bool:
        return "{lines}" in tpl

    if _looks_mojibake(text):
        fixed = _demojibake(text)
        if fixed and ok(fixed):
            logger.warning("[ltm] ltm_prompt.txt 是乱码，已自动还原（建议检查文件编码）")
            text = fixed
        else:
            logger.warning("[ltm] ltm_prompt.txt 是乱码且无法还原，回落到内置提示词")
            return EXTRACT_PROMPT

    if ok(text):
        return text
    logger.warning("[ltm] ltm_prompt.txt 缺少 {lines} 占位符，回落到内置提示词")
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
        self._last_scan_user: dict[str, float] = {}
        # 未被本轮预算处理到的消息留在这里，避免历史游标越过后永久丢失
        self._pending: dict[str, list[str]] = {}
        # 私聊里每个人最近说过的话，让一次单条回复也能带上上下文去抽取
        self._said: dict[str, list[str]] = {}
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
        """Create the facts table and add the newer columns when missing.

        ``dashboard`` also inserts into this table with only the four original
        columns, so every added column must be nullable or carry a default.
        """
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
            existing = {
                row[1] for row in con.execute("PRAGMA table_info(facts)").fetchall()
            }
            for column, ddl in (
                ("slot", "ALTER TABLE facts ADD COLUMN slot TEXT"),
                ("value", "ALTER TABLE facts ADD COLUMN value TEXT"),
                ("source", "ALTER TABLE facts ADD COLUMN source TEXT DEFAULT 'chat'"),
                ("superseded_by", "ALTER TABLE facts ADD COLUMN superseded_by INTEGER"),
                ("conf", "ALTER TABLE facts ADD COLUMN conf REAL DEFAULT 1.0"),
            ):
                if column not in existing:
                    con.execute(ddl)
                    logger.info(f"[ltm] schema: added column {column}")
            con.execute("CREATE INDEX IF NOT EXISTS idx_facts_uid ON facts(uid)")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_live "
                "ON facts(uid, superseded_by)"
            )
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

    async def _chat(self, prompt: str, max_tokens: int | None = None) -> str:
        """Run one short completion against the active provider.

        Reasoning models spend the whole token budget on reasoning and return
        an empty ``content`` when ``max_tokens`` is small, so reasoning is
        explicitly disabled.
        """
        base, key, model = _resolve_llm()
        limit = max_tokens or int(load_settings().get("max_tokens") or 300)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": limit,
            "temperature": 0.2,
            "thinking": {"type": "disabled"},
        }
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
            if resp.status_code in (400, 422):
                # 有些后端不认 thinking 参数，去掉再试一次
                payload.pop("thinking", None)
                resp = await client.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""

    # ---------- recall ----------

    def _recent_rows(self, uid: str, limit: int) -> list[tuple[str, str | None, float]]:
        """Return the newest live facts as ``(text, slot, created)``."""
        con = self._conn()
        try:
            return con.execute(
                "SELECT text, slot, created FROM facts "
                "WHERE uid = ? AND superseded_by IS NULL "
                "ORDER BY created DESC LIMIT ?",
                (uid, limit),
            ).fetchall()
        finally:
            con.close()

    @staticmethod
    def _age_tag(created: float, now: float) -> str:
        """Describe how old a memory is, in the unit people actually use."""
        days = max(0.0, (now - created) / 86400)
        if days < 1:
            return "今天"
        if days < 2:
            return "昨天"
        if days < 7:
            return f"{int(days)}天前"
        if days < 40:
            return f"{int(days // 7)}周前"
        return f"{int(days // 30)}个月前"

    def _timed(self, text: str, slot: str | None, created: float, now: float) -> str:
        """Stamp the memories whose meaning depends on when they were said."""
        if slot in TIMED_SLOTS:
            return f"{text}（{self._age_tag(created, now)}）"
        return text

    def _recent(self, uid: str, limit: int) -> list[str]:
        """Return the most recently stored live facts, time-stamped."""
        now = time.time()
        return [self._timed(t, s, c, now) for t, s, c in self._recent_rows(uid, limit)]

    async def _recall(self, uid: str, query: str) -> list[str]:
        """Pick memories relevant to the current message, no duplicates."""
        now = time.time()
        if not (query or "").strip():
            return [
                self._timed(t, s, c, now)
                for t, s, c in self._recent_rows(uid, MAX_INJECTED)
            ]

        con = self._conn()
        try:
            rows = con.execute(
                "SELECT text, emb, created, slot FROM facts "
                "WHERE uid = ? AND superseded_by IS NULL AND created >= ? "
                "ORDER BY created DESC LIMIT 400",
                (uid, now - 30 * 86400),
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return []

        def already(label: str, picked: list[str]) -> bool:
            """Same text, or nearly same text, is already in the block."""
            norm = self._norm(label)
            return any(self._similar(norm, self._norm(p)) >= 0.85 for p in picked)

        vectors = await self._embed([query])
        if not vectors:
            picked: list[str] = []
            for text, _emb, created, slot in rows[:MAX_INJECTED]:
                label = self._timed(text, slot, created, now)
                if not already(label, picked):
                    picked.append(label)
            return picked

        qv = vectors[0]
        scored: list[tuple[float, str, str | None, float]] = []
        for text, emb, created, slot in rows:
            try:
                fv = json.loads(emb) if emb else None
            except json.JSONDecodeError:
                fv = None
            score = _cosine(qv, fv) if fv else 0.0
            if score >= RECALL_MIN_SCORE:
                scored.append((score, text, slot, created))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = []
        for _score, text, slot, created in scored:
            label = self._timed(text, slot, created, now)
            if already(label, picked):
                continue
            picked.append(label)
            if len(picked) >= RECALL_TOP_K:
                break

        # Always keep a few recent memories so the bot does not forget context.
        for text, slot, created in self._recent_rows(uid, 4):
            label = self._timed(text, slot, created, now)
            if already(label, picked):
                continue
            picked.append(label)

        return picked[:MAX_INJECTED]

    # ---------- extraction ----------

    @staticmethod
    def _norm(text: str) -> str:
        """Lowercase and keep only CJK characters, latin letters and digits."""
        return re.sub(r"[^\u4e00-\u9fff0-9a-z]", "", (text or "").lower())

    @staticmethod
    def _similar(a: str, b: str) -> float:
        """Return a 0..1 similarity ratio between two normalised strings."""
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _known_facts(self, uid: str) -> str:
        """Render the slot summary the model sees, so it stops re-extracting.

        Args:
            uid: Sender id.

        Returns:
            One ``slot: value、value`` per line, or a placeholder line.
        """
        con = self._conn()
        try:
            rows = con.execute(
                "SELECT slot, value, created FROM facts "
                "WHERE uid = ? AND superseded_by IS NULL AND value IS NOT NULL "
                "ORDER BY created DESC LIMIT 80",
                (uid,),
            ).fetchall()
        finally:
            con.close()
        if not rows:
            return "（还没有任何记忆）"
        per_slot: dict[str, list[str]] = {}
        for slot, value, _created in rows:
            bucket = per_slot.setdefault(slot or "misc", [])
            if value not in bucket and len(bucket) < 4:
                bucket.append(value)
        lines = []
        for slot in sorted(per_slot):
            lines.append(f"{slot}: {'、'.join(per_slot[slot])}")
        return "\n".join(lines)

    @staticmethod
    def _worth_sending(text: str) -> bool:
        """Return True when a message may carry a first-person fact.

        This is only a coarse filter that saves prompt tokens: instructions,
        pure emoji and pure questions never reach the model.

        Args:
            text: One raw message.

        Returns:
            Whether the message is worth model tokens.
        """
        t = (text or "").strip()
        if len(t) < 2:
            return False
        if t[0] in "/#!！":
            return False
        # 纯表情、纯符号
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", t):
            return False
        if re.fullmatch(r"\[[^\]]{1,10}\]", t):
            return False
        # 问句问的是别人，不是自己的事实
        if t.rstrip("。.!！~～ 　").endswith(("吗", "呢", "嘛", "?", "？")):
            return False
        return True

    def _parse_facts(self, raw: str, n_lines: int) -> list[tuple[str, str]]:
        """Parse the model's JSON array into validated ``(slot, value)`` pairs.

        Tolerates `````json`` fences and stray prose around the array.

        Args:
            raw: Raw model output.
            n_lines: How many messages were sent (used to sanity-check ``ref``).

        Returns:
            Accepted pairs, in model order.
        """
        m = re.search(r"\[.*\]", raw or "", re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            logger.info("[ltm] model output is not valid JSON, dropped")
            return []
        if not isinstance(data, list):
            return []

        out: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            slot = str(item.get("slot", "")).strip().lower()
            value = str(item.get("value", "")).strip().strip("`\"'“”‘’。，、；,.!！?？~～ 　")
            if slot not in SLOT_WHITELIST:
                continue
            if not 1 <= len(value) <= 24:
                continue
            if value.lower() in USELESS_VALUES:
                continue
            if slot == "relation" and not any(w in value for w in RELATION_WORDS):
                continue
            ref = item.get("ref")
            if isinstance(ref, int) and not 1 <= ref <= max(1, n_lines):
                # 编号越界说明模型在编，这条不要
                logger.info(f"[ltm] drop out-of-range ref {ref} for {slot}={value!r}")
                continue
            key = (slot, self._norm(value))
            if key in seen:
                continue
            seen.add(key)
            out.append((slot, value))
            if len(out) >= MAX_CANDIDATES:
                break
        return out

    @staticmethod
    def _render(slot: str, value: str) -> str:
        """Turn a ``(slot, value)`` into one natural Chinese sentence."""
        v = (value or "").strip()
        if slot == "pet" and not re.search(r"[只条个匹头]", v):
            v = f"只{v}" if len(v) <= 6 else v
        return SLOT_TEMPLATES.get(slot, "{value}").format(value=v)

    async def _extract(self, uid: str, lines: list[str], source: str = "scan") -> None:
        """Send one batch to the model and persist what it finds.

        Args:
            uid: Sender id the facts belong to.
            lines: The user's own messages (a batch, oldest first).
            source: ``chat`` for a private-chat turn, ``scan`` for group history.
        """
        usable = [t for t in lines if self._worth_sending(t)]
        if not usable:
            return
        usable = usable[-EXTRACT_BATCH:]

        listing = "\n".join(f"{i}|{t}" for i, t in enumerate(usable, 1))
        prompt = load_prompt()
        if "{known}" in prompt:
            prompt = prompt.replace("{known}", self._known_facts(uid), 1)
        prompt = prompt.replace("{lines}", listing, 1)

        try:
            raw = await self._chat(prompt)
        except Exception as exc:  # noqa: BLE001 - skip this round, retry later
            logger.warning(f"[ltm] extract call failed ({source}): {exc}")
            return
        if not (raw or "").strip():
            logger.info(f"[ltm] empty model reply for {uid} ({source})")
            return

        tags = self._parse_facts(raw, len(usable))
        if not tags:
            logger.info(
                f"[ltm] nothing new for {uid} ({source}, {len(usable)} msgs, "
                f"{len(raw)} chars reply)"
            )
            return

        produced: list[tuple[str, str, str]] = []
        for slot, value in tags:
            text = self._render(slot, value)
            text = re.sub(r"\s+", "", text).strip("。，、；;,.!")
            if not 4 <= len(text) <= MAX_FACT_CHARS:
                continue
            produced.append((slot, value, text))
        if not produced:
            return

        await self._store(uid, produced, source)

    async def _store(
        self, uid: str, produced: list[tuple[str, str, str]], source: str
    ) -> None:
        """Persist facts with slot-keyed dedupe.

        Single-value slots keep one live row per slot: a new value supersedes
        the old one outright, which makes wording drift unable to pile up
        duplicates. Multi-value slots dedupe by normalised value, text
        similarity and embedding similarity, then cap per-slot rows FIFO.
        """
        texts = [text for _s, _v, text in produced]
        vectors = await self._embed(texts) or []

        con = self._conn()
        added = 0
        try:
            existing = con.execute(
                "SELECT id, text, emb, slot, value, created FROM facts "
                "WHERE uid = ? AND superseded_by IS NULL "
                "ORDER BY created DESC LIMIT 300",
                (uid,),
            ).fetchall()
            live = list(existing)
            now = time.time()

            for idx, (slot, value, text) in enumerate(produced):
                vec = vectors[idx] if idx < len(vectors) else []
                nv = self._norm(value)
                norm_text = self._norm(text)
                dup = False
                stale: list[int] = []

                if slot in SINGLE_SLOTS:
                    for fid, old_text, _e, old_slot, old_value, _c in live:
                        if old_slot != slot:
                            continue
                        ov = self._norm(old_value or old_text)
                        if ov == nv or self._similar(ov, nv) >= 0.85:
                            dup = True
                            break
                        stale.append(fid)  # 同槽位旧值一律被新值顶掉
                else:
                    for fid, old_text, old_emb, old_slot, old_value, _c in live:
                        if old_slot != slot:
                            continue
                        ov = self._norm(old_value or "")
                        if ov and nv and (ov == nv or self._similar(ov, nv) >= DEDUP_TEXT):
                            dup = True
                            break
                        if self._similar(norm_text, self._norm(old_text)) >= DEDUP_TEXT:
                            dup = True
                            break
                        if vec and old_emb:
                            try:
                                ovec = json.loads(old_emb)
                            except json.JSONDecodeError:
                                ovec = None
                            if ovec and _cosine(vec, ovec) >= DEDUP_EMB:
                                dup = True
                                break

                if dup:
                    continue

                cur = con.execute(
                    "INSERT INTO facts (uid, text, emb, created, slot, value, source, conf) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        uid,
                        text,
                        json.dumps(vec) if vec else None,
                        now,
                        slot,
                        value,
                        source,
                        1.0,
                    ),
                )
                new_id = cur.lastrowid
                for fid in stale:
                    con.execute(
                        "UPDATE facts SET superseded_by = ? "
                        "WHERE id = ? AND superseded_by IS NULL",
                        (new_id, fid),
                    )
                live.append(
                    (new_id, text, json.dumps(vec) if vec else None, slot, value, now)
                )
                added += 1

            # 多值槽位超上限：把最旧的压掉
            counts: dict[str, list[int]] = {}
            for fid, _t, _e, old_slot, _v, _c in live:
                if old_slot in MULTI_SLOTS:
                    counts.setdefault(old_slot, []).append(fid)
            for slot_ids in counts.values():
                for fid in slot_ids[:-MULTI_SLOT_CAP] if len(slot_ids) > MULTI_SLOT_CAP else []:
                    con.execute(
                        "UPDATE facts SET superseded_by = -1 WHERE id = ?", (fid,)
                    )

            con.execute(
                """DELETE FROM facts WHERE uid = ? AND id NOT IN (
                       SELECT id FROM facts WHERE uid = ? ORDER BY created DESC LIMIT ?)""",
                (uid, uid, MAX_FACTS_PER_USER),
            )
            con.commit()
            if added:
                logger.info(f"[ltm] stored {added} new fact(s) for {uid} ({source})")
        finally:
            con.close()

    # ---------- hooks ----------

    @filter.on_llm_request()
    async def inject_memory(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Append relevant memories to the system prompt.

        The block is capped twice on purpose: one line per memory (60 chars)
        and 600 chars overall, roughly 0.6% of a 64K window.
        """
        uid = event.get_sender_id() or event.get_session_id()
        if not uid:
            return
        query = (event.message_str or "").strip()
        memories = await self._recall(uid, query)
        if not memories:
            return
        lines = []
        budget = INJECT_MAX_CHARS
        for mem in memories:
            line = mem if len(mem) <= INJECT_LINE_CHARS else mem[:INJECT_LINE_CHARS] + "…"
            if budget - len(line) < 0:
                break
            if any(line in existing or existing in line for existing in lines):
                continue
            lines.append(line)
            budget -= len(line) + 2
        if not lines:
            return
        block = "\n".join(lines)
        req.system_prompt = (req.system_prompt or "") + (
            "\n\n[你长期记得的关于对方的事，自然地用，不要逐条复述]\n" + block
        )

    def _append_said(self, uid: str, texts: list[str]) -> list[str]:
        """Record what one person just said and return the rolling window."""
        buf = self._said.setdefault(uid, [])
        buf.extend(t for t in texts if t and t.strip())
        if len(buf) > 24:
            del buf[:-24]
        return buf

    @filter.on_llm_response()
    async def learn_from_turn(
        self, event: AstrMessageEvent, response: LLMResponse
    ) -> None:
        """Schedule fact extraction for private chats only.

        Group messages are all captured by the background scanner (they are in
        ``platform_message_history``), so replying in a group never triggers
        an extra model call here. Private chats have no scanner, so a reply is
        the only chance to learn — throttled by ``extract_cooldown``.
        """
        uid = event.get_sender_id() or event.get_session_id()
        if not uid:
            return
        # 群聊一律交给扫描器攒批处理，回复不触发
        if event.get_group_id():
            return
        user_text = (event.message_str or "").strip()
        if len(user_text) < 4:
            return

        said = self._append_said(uid, [user_text])

        now = time.time()
        cooldown = float(load_settings().get("extract_cooldown") or 900)
        if now - self._last_extract.get(uid, 0.0) < cooldown:
            return
        self._last_extract[uid] = now

        asyncio.create_task(self._extract(uid, list(said), "chat"))

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
        """Periodically accumulate group messages and extract in batches."""
        while True:
            conf = load_settings()
            interval = max(60, int(conf.get("scan_interval") or 300))
            await asyncio.sleep(interval)
            if not conf.get("enabled", True):
                continue
            try:
                await self._scan_once()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                logger.warning(f"[ltm] scan failed: {exc}")

    def _is_third_party(self) -> bool:
        """Return True when the bot runs on the third-party API."""
        try:
            cfg = json.loads(CMD_CONFIG.read_text(encoding="utf-8-sig"))
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
        state = DATA_DIR / "api_mode.json"
        if not state.exists():
            return False
        try:
            return json.loads(state.read_text(encoding="utf-8")).get("mode") == "third_party"
        except Exception:  # noqa: BLE001
            return False

    async def _scan_once(self) -> None:
        """Queue new group messages per sender and extract full batches only.

        Waking up is cheap (no model call). A person goes to the model only
        when ``MIN_BATCH`` usable messages piled up, or when pending messages
        have waited longer than ``MAX_WAIT`` — one call then digests
        everything that person said since the last extraction.
        """
        conf = load_settings()
        if self._is_third_party() and not conf.get("third_party_scan", True):
            logger.info("[ltm] third-party mode: memory scan disabled by config")
            return

        rows = self._new_group_rows()
        if rows:
            self._last_hist_id = rows[-1][0]
            for _rid, uid, text in rows:
                buf = self._pending.setdefault(uid, [])
                buf.append(text)
                # 每人只留最近 60 条，控制内存
                if len(buf) > 60:
                    del buf[:-60]

        if not self._pending:
            return

        budget = max(1, int(conf.get("scan_budget") or 3))
        cooldown = float(conf.get("user_cooldown") or 1800)
        now = time.time()
        picked: list[tuple[int, str, list[str]]] = []
        for uid, texts in self._pending.items():
            usable = [t for t in texts if self._worth_sending(t)]
            if not usable:
                self._pending[uid] = []
                continue
            if now - self._last_scan_user.get(uid, 0.0) < cooldown:
                continue  # 冷却中，保留 pending 等下一轮
            last = self._last_scan_user.get(uid, 0.0)
            waited = now - last
            if len(usable) >= MIN_BATCH or waited >= MAX_WAIT:
                picked.append((len(usable), uid, usable))

        picked.sort(key=lambda item: -item[0])

        handled = 0
        for _score, uid, usable in picked:
            if budget <= 0:
                break
            budget -= 1
            handled += 1
            self._last_scan_user[uid] = now
            batch = usable[:EXTRACT_BATCH]
            self._pending[uid] = usable[EXTRACT_BATCH:]
            await self._extract(uid, batch, "scan")

        for uid in [k for k, v in self._pending.items() if not v]:
            self._pending.pop(uid, None)
        backlog = sum(len(v) for v in self._pending.values())
        logger.info(
            f"[ltm] scanned {len(rows)} msgs, {handled} sender(s) extracted, "
            f"{backlog} msg(s) queued"
        )

    # ---------- commands ----------

    @filter.command("记忆")
    async def show_memory(self, event: AstrMessageEvent) -> None:
        """List what the bot remembers about the sender."""
        uid = event.get_sender_id() or event.get_session_id()
        con = self._conn()
        try:
            rows = con.execute(
                "SELECT text, slot, created, superseded_by FROM facts "
                "WHERE uid = ? ORDER BY created DESC LIMIT 20",
                (uid,),
            ).fetchall()
        finally:
            con.close()
        if not rows:
            yield event.plain_result("我还没记住关于你的事。")
            return
        now = time.time()
        body = "\n".join(
            f"{i}. {self._timed(text, slot, created, now)}"
            + ("（已过期）" if superseded else "")
            for i, (text, slot, created, superseded) in enumerate(rows, 1)
        )
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
