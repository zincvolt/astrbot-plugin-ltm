"""Long-term memory for QQ chats.

Extracts durable facts about each sender after a reply, stores them with
embeddings, and injects the most relevant memories back into the system
prompt before the next LLM request.

The plugin never touches the LLM deployment itself: it only calls the
OpenAI-compatible endpoints already running on this machine.
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
    "scan_interval": 300,      # 多久扫一次群历史（秒）
    "scan_budget": 3,          # 每轮最多为几个人抽取
    "user_cooldown": 900,      # 同一个人两次扫描抽取的最小间隔（秒）
    "extract_cooldown": 45,    # 一次对话回复后触发的抽取冷却（秒）
    "max_tokens": 500,
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
EXTRACT_COOLDOWN = 45
MAX_FACT_CHARS = 120
MAX_FACTS_PER_USER = 300
DEDUP_SCORE = 0.92
MAX_CANDIDATES = 6          # 一次最多改写几个片段，控制小模型负担

# 改写结果里出现这些词说明模型在猜，不是原话
HALLUCINATION_MARKERS = ("可能", "也许", "好像", "估计", "应该", "大概", "似乎")

PROMPT_FILE = Path(os.environ.get("LTM_PROMPT", str(DATA_DIR / "ltm_prompt.txt")))

# (mtime, settings) / (mtime, (base, key, model)) — 配置文件改动后自动失效
_settings_cache: tuple[float, dict[str, Any]] | None = None
_llm_cache: tuple[float, tuple[str, str, str]] | None = None


def _resolve_llm() -> tuple[str, str, str]:
    """Resolve the chat endpoint of the provider AstrBot is currently using.

    Memory extraction used to hard-code the local 8080 server, which silently
    fails in third-party mode because that server is stopped. Follow whatever
    ``agent_runner`` points at instead.

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

# ---------------------------------------------------------------------------
# 抽取算法：规则挖候选 -> 模型只做改写
#
# 先用正则从原话里挖出"明确的第一人称断言"片段，模型不再负责判断"记什么"，
# 只负责把片段润成一句第三人称陈述。小模型做这么小的事才稳；模型挂了也能用
# 模板兜底，保证有候选就有记忆。
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """把下面每个片段改写成一句关于「对方」的第三人称陈述。

规则：
- 每行输出一个，格式：编号=陈述
- 只改写片段里已有的内容，不许添加片段里没有的信息
- 陈述不超过 20 字，用「对方」开头
- 片段看不懂、不完整、或只是当下的状态，就输出：编号=NO
- 不要解释，不要空行

示例：
1=喜欢猫 -> 1=对方喜欢猫
2=今天好累 -> 2=NO

待改写片段：
{candidates}
"""

# 槽位 -> 优先级；正则第 1 组是核心值
CANDIDATE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("name", re.compile(r"叫我\s*([^\s，。！？、,.!?~～]{1,10})")),
    ("name", re.compile(r"我(?:的名字)?叫\s*([^\s，。！？、,.!?~～]{1,10})")),
    ("age", re.compile(r"我(?:今年)?\s*(\d{1,2})\s*(?:岁|周岁)")),
    (
        "location",
        re.compile(
            r"我(?:现在|目前)?(?:住在|住|在)\s*([\u4e00-\u9fa5A-Za-z]{2,10}?)(?:市|区|县|附近|这边|城里)"
        ),
    ),
    ("location", re.compile(r"我(?:是|来自)\s*([\u4e00-\u9fa5]{2,8})(?:人|本地)")),
    # 「我是」后面太杂，正则只负责抓，身份词由 _is_identity 校验
    ("identity", re.compile(r"我是(?:一个|一名|个)?\s*([\u4e00-\u9fa5A-Za-z]{2,12})")),
    ("like", re.compile(r"我(?:最|很|超|特别|蛮|挺)?(?:喜欢|爱|迷上|厨)\s*([^\s，。！？、,.!?~～]{1,16})")),
    ("dislike", re.compile(r"我(?:不喜欢|讨厌|受不了|怕|不玩|不吃)\s*([^\s，。！？、,.!?~～]{1,16})")),
    ("habit", re.compile(r"我(?:每天|每晚|经常|一般|通常|习惯|老是|总是)\s*([^\s，。！？、,.!?~～]{2,18})")),
    (
        "pet",
        re.compile(
            r"我(?:养|有)了?\s*(?:一只|一个|一条|两只|三只)?\s*"
            r"([\u4e00-\u9fa5]{1,8}?(?:猫|狗|兔|鸟|仓鼠|蛇|蜥蜴|龙猫|鱼|鹦鹉))"
        ),
    ),
    (
        "relation",
        re.compile(
            r"我(?:对象|女朋友|男朋友|老婆|老公|室友|哥哥|姐姐|弟弟|妹妹)(?:是|在|叫)\s*"
            r"([^\s，。！？、,.!?~～]{1,14})"
        ),
    ),
    # 「我在做某事」，要求带「在」且后面不是形容词性状态
    ("activity", re.compile(r"我(?:最近|这阵子)?在\s*([^\s，。！？、,.!?~～]{2,16})")),
]

# 身份词：只有以这些词结尾的「我是X」才算长期身份
IDENTITY_SUFFIXES = (
    "学生", "大学生", "高中生", "研究生", "程序员", "工程师", "画师", "设计师",
    "老师", "医生", "护士", "律师", "会计", "兽人", "社畜", "玩家", "写手",
    "乐手", "作者", "主播", "店主", "公务员", "研究生",
)
# 值以这些结尾说明在指代眼前的东西，记下来没意义
VAGUE_SUFFIXES = ("这个", "那个", "这些", "那些", "这种", "那种", "这里", "那里")
# 值开头这些虚词是噪音：「我每天都喝咖啡」不该记成「经常都喝咖啡」
VALUE_PREFIX_NOISE = ("都", "就", "也", "还", "又")

# 这些词说明说的是"眼下这一刻"，不是长期事实
TEMPORAL_FLAGS = (
    "今天", "今晚", "刚才", "刚刚", "现在", "这会儿", "马上", "明天",
    "昨天", "昨晚", "等下", "待会", "这两天",
)
# 命中这些槽位时，带时间副词就更可能是临时状态，直接丢
VOLATILE_SLOTS = {"like", "dislike", "habit"}

# 值本身不能是这些东西
USELESS_VALUES = {
    "你", "他", "她", "它", "你们", "他们", "她们", "这个", "那个", "这些",
    "那些", "什么", "怎么", "这样", "那样", "一下", "一点", "真的", "反正",
    "哈哈", "呵呵", "东西", "事情", "问题", "感觉",
}

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
    "activity": "对方在{value}",
}


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
    """Return the rewrite prompt, preferring the editable file.

    Recovers from the common "UTF-8 saved as Latin-1" mojibake that silently
    turns the whole prompt into gibberish for the model.

    Returns:
        Prompt template containing the ``{candidates}`` placeholder.
    """
    try:
        text = PROMPT_FILE.read_text(encoding="utf-8")
    except OSError:
        return EXTRACT_PROMPT

    if _looks_mojibake(text):
        fixed = _demojibake(text)
        if fixed and "{candidates}" in fixed:
            logger.warning("[ltm] ltm_prompt.txt 是乱码，已自动还原（建议检查文件编码）")
            text = fixed
        else:
            logger.warning("[ltm] ltm_prompt.txt 是乱码且无法还原，回落到内置提示词")
            return EXTRACT_PROMPT

    if "{candidates}" in text:
        return text
    logger.warning("[ltm] ltm_prompt.txt 缺少 {candidates} 占位符，回落到内置提示词")
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

        Thinking models (DeepSeek V4 and friends) spend the whole token budget
        on reasoning and return an empty ``content`` when ``max_tokens`` is
        small, so reasoning is explicitly disabled.
        """
        base, key, model = _resolve_llm()
        limit = max_tokens or int(load_settings().get("max_tokens") or 500)
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

    def _recent(self, uid: str, limit: int) -> list[str]:
        """Return the most recently stored live facts for a user."""
        con = self._conn()
        try:
            rows = con.execute(
                "SELECT text FROM facts WHERE uid = ? AND superseded_by IS NULL "
                "ORDER BY created DESC LIMIT ?",
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
                "SELECT id, text, emb, created FROM facts "
                "WHERE uid = ? AND superseded_by IS NULL AND created >= ? "
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

    def _mine(self, text: str) -> list[tuple[str, str]]:
        """Mine high-confidence ``(slot, value)`` pairs from one message.

        Only explicit first-person statements are taken. Anything that looks
        like a question, a one-off state or a pronoun is dropped: a missing
        memory is cheaper than a wrong one.

        Args:
            text: One raw message from the user.

        Returns:
            Slot/value pairs in rule-priority order.
        """
        cleaned = (text or "").strip()
        if len(cleaned) < 4:
            return []

        has_volatile = any(word in cleaned for word in TEMPORAL_FLAGS)
        found: list[tuple[str, str]] = []
        per_slot: dict[str, list[str]] = {}

        for slot, pattern in CANDIDATE_RULES:
            if slot in VOLATILE_SLOTS and has_volatile:
                continue
            for m in pattern.finditer(cleaned):
                # 「我喜欢猫吗」这类是问句，不是陈述
                if re.match(r"^\s*(吗|呢|嘛|么|？|\?)", cleaned[m.end():]):
                    continue
                value = m.group(1).strip(" 　的了吧啊呀哦啦")
                for prefix in VALUE_PREFIX_NOISE:
                    if value.startswith(prefix) and len(value) > 2:
                        value = value[1:]
                        break
                value = value.strip(" 　的了吧啊呀哦啦")
                if not value or value in USELESS_VALUES:
                    continue
                if value.endswith(VAGUE_SUFFIXES):
                    continue
                if re.search(r"[吗呢嘛]$", value):
                    continue
                if slot == "identity" and not value.endswith(IDENTITY_SUFFIXES):
                    continue
                # 单字只对「喜欢/讨厌/养」这类成立，猫、狗、鱼都是好事实
                lo = 1 if slot in ("name", "age", "like", "dislike", "pet") else 2
                hi = 10 if slot in ("name", "age") else 16
                if not lo <= len(value) <= hi:
                    continue
                bucket = per_slot.setdefault(slot, [])
                if value in bucket:
                    continue
                bucket.append(value)
                found.append((slot, value))
                if len(bucket) >= 2:
                    break

        return found[:MAX_CANDIDATES]

    async def _rewrite(self, cands: list[tuple[str, str]]) -> dict[int, str]:
        """Ask the model to turn mined snippets into third-person statements.

        The model only sees a numbered snippet list, so the task stays small
        enough for the local 9B model. Any failure returns an empty mapping and
        the caller falls back to templates.

        Args:
            cands: Mined slot/value pairs.

        Returns:
            Mapping of 1-based snippet index to rewritten sentence.
        """
        listing = "\n".join(
            f"{i}={value}" for i, (_slot, value) in enumerate(cands, 1)
        )
        try:
            raw = await self._chat(load_prompt().replace("{candidates}", listing))
        except Exception as exc:  # noqa: BLE001 - templates keep extraction alive
            logger.warning(f"[ltm] rewrite call failed, 用模板兜底: {exc}")
            return {}

        out: dict[int, str] = {}
        for line in raw.splitlines():
            m = re.match(r"^\s*(\d+)\s*[=:：]\s*(.+?)\s*$", line)
            if not m:
                continue
            body = m.group(2).strip().strip("`\"'“”‘’")
            if body.upper() in ("NO", "N/A", "无", "跳过", "无。"):
                continue
            out[int(m.group(1))] = body
        return out

    def _keeps_core(self, text: str, value: str) -> bool:
        """Return True when a rewritten sentence still carries the mined value.

        A literal substring test is too strict for a 9B model ("都要喝咖啡" ->
        "爱喝咖啡" is a fair paraphrase), so this compares character sets and
        only rejects rewrites that dropped most of the value.

        Args:
            text: Candidate rewritten sentence.
            value: Original snippet that must survive the rewrite.

        Returns:
            Whether enough of the snippet is present.
        """
        core = set(self._norm(value))
        if not core:
            return True
        return len(core & set(self._norm(text))) / len(core) >= 0.5

    async def _extract(
        self, uid: str, lines: list[str], source: str = "chat"
    ) -> None:
        """Mine, rewrite and persist facts for one user.

        Args:
            uid: Sender id the facts belong to.
            lines: The user's own messages (one turn, or a batch from a group).
            source: ``chat`` for a direct turn, ``scan`` for group-history scan.
        """
        cands: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for line in lines:
            for slot, value in self._mine(line):
                key = (slot, self._norm(value))
                if key in seen:
                    continue
                seen.add(key)
                cands.append((slot, value))
                if len(cands) >= MAX_CANDIDATES:
                    break
            if len(cands) >= MAX_CANDIDATES:
                break
        if not cands:
            return

        rewritten = await self._rewrite(cands)

        produced: list[tuple[str, str, str]] = []
        for idx, (slot, value) in enumerate(cands, 1):
            text = rewritten.get(idx, "")
            # 模型改写不能把核心值丢掉，否则视为幻觉，退回模板
            if not text or not self._keeps_core(text, value):
                text = SLOT_TEMPLATES.get(slot, "").format(value=value)
            text = re.sub(r"\s+", "", text).strip("。，、；;,.!")
            if not 4 <= len(text) <= MAX_FACT_CHARS:
                continue
            if any(word in text for word in HALLUCINATION_MARKERS):
                continue
            produced.append((slot, value, text))
        if not produced:
            return

        await self._store(uid, produced, source)

    async def _store(
        self, uid: str, produced: list[tuple[str, str, str]], source: str
    ) -> None:
        """Persist new facts, skipping duplicates and retiring weaker wording.

        Duplicate rules, in order: identical text, identical value, one value
        containing the other (same slot), embedding similarity. When a shorter
        wording is replaced by a fuller one the old row is marked
        ``superseded_by`` instead of being deleted.
        """
        vectors = await self._embed([text for _s, _v, text in produced]) or []

        con = self._conn()
        added = 0
        try:
            existing = con.execute(
                "SELECT id, text, emb, slot, value FROM facts "
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

                for fid, old_text, old_emb, old_slot, old_value in live:
                    if norm_text == self._norm(old_text):
                        dup = True
                        break
                    ov = self._norm(old_value or "")
                    if ov and nv:
                        if ov == nv:
                            dup = True
                            break
                        if old_slot == slot and (ov in nv or nv in ov):
                            stale.append(fid)
                            continue
                    if vec and old_emb:
                        try:
                            ovec = json.loads(old_emb)
                        except json.JSONDecodeError:
                            ovec = None
                        if ovec and _cosine(vec, ovec) >= DEDUP_SCORE:
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
                live.append((new_id, text, json.dumps(vec) if vec else None, slot, value))
                added += 1

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
        if not uid or len(user_text) < 6:
            return

        now = time.time()
        cooldown = float(load_settings().get("extract_cooldown") or EXTRACT_COOLDOWN)
        if now - self._last_extract.get(uid, 0.0) < cooldown:
            return
        self._last_extract[uid] = now

        asyncio.create_task(self._extract(uid, [user_text], "chat"))

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
            conf = load_settings()
            interval = max(30, int(conf.get("scan_interval") or 300))
            await asyncio.sleep(interval)
            if not conf.get("enabled", True):
                continue
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
        """Queue new group messages per sender and extract for a few of them."""
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
                # 每人只留最近 40 条，控制内存
                if len(buf) > 40:
                    del buf[:-40]

        if not self._pending:
            return

        budget = max(1, int(conf.get("scan_budget") or 3))
        cooldown = float(conf.get("user_cooldown") or 900)
        now = time.time()
        # 先用规则过一遍：挖不出候选的人不值得花一次模型调用
        scored: list[tuple[int, str, list[str]]] = []
        for uid, texts in self._pending.items():
            if not texts:
                continue
            if now - self._last_scan_user.get(uid, 0.0) < cooldown:
                continue  # 冷却中，保留 pending 等下一轮
            hits = sum(len(self._mine(t)) for t in texts)
            if hits:
                scored.append((hits, uid, texts))
        # 候选多的人优先，信息量通常更大
        scored.sort(key=lambda item: -item[0])

        handled = 0
        for _hits, uid, texts in scored:
            if budget <= 0:
                break
            budget -= 1
            handled += 1
            self._last_scan_user[uid] = now
            self._pending[uid] = []
            await self._extract(uid, texts, "scan")

        for uid in [k for k, v in self._pending.items() if not v]:
            self._pending.pop(uid, None)
        logger.info(
            f"[ltm] scanned {len(rows)} msgs, {handled} sender(s) extracted, "
            f"{len(self._pending)} pending"
        )

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
