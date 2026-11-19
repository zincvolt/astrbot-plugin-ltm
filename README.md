# astrbot-plugin-ltm

QQ 群聊的长期记忆插件，用于 [AstrBot](https://github.com/AstrBotDevs/AstrBot)。

它让机器人能"记住"跨会话的稳定事实：每轮回复后从对话里抽取值得记的内容，存成带向量的记忆；下次请求前按当前消息语义召回最相关的几条，注入 system prompt。

## 功能

- **记忆抽取**：回复后异步从 `用户发言 + 机器人回复` 中抽取稳定事实（人物、偏好、约定等），写入 SQLite + 向量库。
- **语义召回**：`on_llm_request` 钩子按当前消息向量检索相关记忆注入上下文，同一条事实不会反复刷屏。
- **后台扫描**：定时任务批量处理积压的新群消息，只对"挖得出候选"的人调用模型，省 API 费用。
- **第三方模型适配**：会自动跟随 AstrBot 当前选用的 provider（`cmd_config.json` 里 `agent_runner` 的配置），处于非本地模型 API 模式时不会去连已经停掉的本地端口。
- **管理命令**：`/记忆` 查看当前用户记忆，`/忘记 [关键词]` 删除记忆。

## 抽取算法

不用"把对话丢给模型，让它输出 JSON 事实"，那条路对本地 9B 模型太难，实测几乎抽不出东西。这里改成了两级：

```
原话 ──[1] 规则挖候选 ──> 候选片段 ──[2] 模型只做改写 ──> 一句第三人称陈述
                │
                └─ 模型不可用 / 输出不合规 ──> 模板兜底，保证有候选就有记忆
```

**第一级：规则挖候选（纯代码，零成本）**

用一组正则匹配明确的第一人称断言，例如：

| 槽位 | 命中示例 | 抽出值 |
|---|---|---|
| `name` | 我叫小灰 / 叫我灰灰 | 小灰 |
| `age` | 我今年 19 岁 | 19 |
| `location` | 我是南京人 / 我住在苏州 | 南京 / 苏州 |
| `identity` | 我是大学生 | 大学生 |
| `like` | 我喜欢打游戏 | 打游戏 |
| `dislike` | 我讨厌下雨天 | 下雨天 |
| `habit` | 我每天都喝咖啡 | 喝咖啡 |
| `pet` | 我养了一只仓鼠 | 一只仓鼠 |
| `relation` | 我室友叫阿明 | 阿明 |
| `activity` | 我在南京上班 | 南京上班 |

同时会丢掉：问句（`我喜欢猫吗`）、带时间副词的一次性状态（`今天我喜欢下雨`）、指代不明的值（`我喜欢吃这个`）、不是长期身份的 `我是X`（`我是个好脾气`）。

**第二级：模型只做改写**

模型拿到的是编号片段列表，任务只有一句："把片段改写成关于对方的第三人称陈述"，输出 `1=对方喜欢打游戏`。任务足够小，小模型也能做对。

硬校验：改写结果必须保留原片段 50% 以上的字符，否则判定为幻觉，改用模板（`对方喜欢{value}`）。模型超时、报错、输出乱码，全部走模板，不会一条都记不住。

**写入去重**

- 文本完全相同 → 跳过
- 同一槽位、值相同 → 跳过
- 同一槽位、一个值是另一个的子串（`猫` / `猫咪`）→ 新说法入库，旧记录标记 `superseded_by` 作废，不再注入
- 向量相似度 ≥ 0.92 → 跳过
- 不同偏好（`喜欢猫` / `喜欢狗`）各自保留，不会互相覆盖

## 提示词

内置在 `main.py` 的 `EXTRACT_PROMPT`，也可以放到 `<数据目录>/ltm_prompt.txt` 覆盖（按 mtime 热读，改完即生效，格式见 `ltm_prompt.example.txt`）。

文件必须含 `{candidates}` 占位符；缺占位符或编码损坏时自动回落到内置提示词。

## 配置

`<数据目录>/ltm.json`，缺省值：

```json
{
  "enabled": true,
  "scan_interval": 300,
  "scan_budget": 3,
  "user_cooldown": 900,
  "extract_cooldown": 45,
  "max_tokens": 500,
  "third_party_scan": true
}
```

`scan_budget` 是每轮最多为几个人抽取；`third_party_scan` 为 false 时，非本地模型 API 模式下不扫群历史（不产生 API 费用）。

## 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `QQNTBOT_ROOT` | 项目根目录 | 自动从 `__file__` 向上推断 |
| `QQNTBOT_DATA` | 数据目录（放 `memory.db`、`ltm.json`） | `<root>/data` |
| `LTM_BOT_DB` | AstrBot 的 `data_v4.db`（群历史来源） | `<root>/apps/AstrBot/data/data_v4.db` |
| `LTM_CMD_CONFIG` | AstrBot 的 `cmd_config.json`（读当前 provider） | `<root>/apps/AstrBot/data/cmd_config.json` |
| `LTM_PROMPT` | 提示词文件路径 | `<数据目录>/ltm_prompt.txt` |
| `LTM_EMB_BASE` | embedding 服务地址 | `http://127.0.0.1:8097/v1` |
| `LTM_EMB_MODEL` | embedding 模型名 | `bge-small-zh-v1.5` |

## 存储

记忆库是运行数据（SQLite `memory.db`），**不随代码发布**，运行时自建。首次启动会在旧库上自动补列（`slot` / `value` / `source` / `superseded_by` / `conf`），旧数据继续可用。

## 依赖

- AstrBot 4.28+
- OpenAI 兼容的 embedding 服务（模型名默认 `bge-small-zh-v1.5`，地址与模型可用 `LTM_EMB_BASE` / `LTM_EMB_MODEL` 覆盖）
- 一个 OpenAI 兼容的 chat 服务用于改写（自动读当前 provider）
- `httpx`

> 只调用外部 OpenAI 兼容接口，不负责部署模型本身。embedding 不可用时会退化成"取最近几条记忆"，不会报错。

## 许可证

AGPL-3.0-or-later，见 [LICENSE](LICENSE)。
