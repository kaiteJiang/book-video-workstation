# 小节计划格式

JSON 顶层字段：`schema_version: "narrative-unit-v1"`、`source_sha256`（正文 UTF-8 内容哈希）、`units`。

每个 unit 包含：

- `id`：唯一小节 ID。
- `start` / `end`：正文字符跨度，含标点和换行；所有小节无间隙、不重叠地覆盖全文。
- `beat`：这个小节发生了什么。
- `before` / `after`：各含非空字符串 `time`、`place`、`state`、`image`。分别写 A/B；不是统一镜头模板。
- `events`：正文实际讲述的中间发展，非空字符串列表。
- `change_kind`：`situation`、`relationship`、`goal`、`consequence` 中的一项；没有 gesture 类型。
- `state_change`：具体说明两端处境为何不同。
- `story_elapsed`：有依据的剧情时间跨度，允许“时间未明，经历完整事件”，不捏造日期。
- `evidence`：资料文件/来源链接/证据 ID，非空列表。人工核验其确实支持正文。
- `b_entry_text`：本节正文中唯一出现、开始揭晓 B 结果的准确原文。
- `next_link`：本节结果如何引出下一节；结尾说明回扣什么。
- `semantic_review`：人工语义复核的具体理由，说明不是微动作，也不是无关名场面硬配对。不得伪称用户已批准。

自动检查只判断完整性、正文哈希、跨度、出场文本和声明的变化类型。人工仍须核查“中间事件→状态变化”真实成立、证据可用和剧情值得讲。
