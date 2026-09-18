> 历史实施计划，不是当前生产指令。AB 语义、视角和批准规则以 `AGENTS.md` 与 `docs/runbooks/longform-story-production.md` 为准；下文旧的单动作和组数建议已被替代。

# 长篇故事升级（用户 2026-09-08 已确认）

目标：完成可复用长篇流程与《万历十五年》《解忧杂货店》两篇长稿待审。禁止本阶段 TTS、生图、ASR、成片或上传；禁止使用去 AI 味 Skill。

## 交付与接口

1. 故事文稿路径：人物第一视角优先，稳定第三视角可选；事件、认知边界、证据、原著引语与改写区分；无需 human-writing/checker。候选、复查和批准必须绑定具体文本及事实材料，仍停在 awaiting_script_review。
2. 长篇 profile：建议 6–8 分钟，10 分钟软上限；可短可长，超 10 分钟产生可审查提醒，不能强制凑字数。旧 profile 兼容。动态 AB 推荐 12–30 组，具体按语义而非匀分；有效数量上限统一并保留安全界限。
3. AB：继续 A 当前哈希作 B 参考、单动作推进、ASR 转折、45 帧墨迹和 15 帧转场。全片覆盖连续无缝；跨组人物/场景参考；3 组代表图可从动态场景选择，长篇语义规划输入支持精彩节点而非机械均分。
4. 声音池：8–12 个官方可核验候选（不伪称试听通过或最热门），用途/性别/听感标签、模型/接口适用性、热度证据与日期、试听待定状态；按故事筛选最多 3 个，代表片段可非开头，返回字符跨度及哈希。
5. 文稿：两篇候选 v2，约 1500–2200 字起步但故事决定长度；补核关键情节，完整因果与首尾照应；不虚构原话、不改旧稿、无媒体生成。附来源和选段理由。
6. 验收：旧流程测试 + 新故事批准/失效测试 + 动态 AB 全覆盖/代表图/哈希测试 + 声音筛选/选段测试；合成测试使用本地 fixture，不调用媒体服务。

## 工作分工

- 内容 agent：content 故事模块、content_runtime/相关 CLI 接口及其测试；不编辑 production/profile、voice、illustration、media_runtime。
- 视觉 agent：production/profile、illustration、media_runtime/必要 media_stages、视觉测试与视觉合同。
- 声音 agent：voice、声音目录数据、声音测试；不编辑 production/profile、media_runtime 或 CLI。
- 主 agent：来源核验、两篇撰写、总集成、文档与独立复核。

## 初始检查

| 接口 | 约定 |
|---|---|
| 内容与视觉 profile | 增加可选 narrative_mode=story；旧默认保持 value；软时长提示不伪装 hard pass |
| 声音与媒体运行 | 保留旧 prepare_candidates 调用签名兼容，新选段参数可选；主 agent 连接 |
| 内容与现有待审稿 | 提供清晰的 candidate import/review 路径，候选不自动批准 |
| 视觉与渲染 | 现有媒体语义与呈现参数保持，数量动态化；测试覆盖旧 3/4 组 |

Ruling: 隔离 worktree 从当前 HEAD 带入现有 tracked 改动与必要 untracked 代码；不暂存或提交用户原有改动。验证完成后以原文件哈希不变为条件同步本次增量回主工作区。
Ruling: 用户要求不用去 AI 味 Skill，高于旧代码和流程中强制 human-writing 的约定；新故事路径必须有真实的直接写作记录，禁止伪造 checker pass。
Ruling: 本轮声音池为官方候选目录与选择能力，不产生试听；实际听感验收在稿件批准后的试听门禁完成。
