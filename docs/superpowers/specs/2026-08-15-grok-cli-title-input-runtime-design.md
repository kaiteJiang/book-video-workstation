# BV Workstation V0.1 增量设计：书名输入、Human Writing 与 Grok CLI 视频

日期：2026-08-15  
状态：已获方向批准，待书面复核  
适用仓库：`BV_Workstation` 0.1  

## 1. 本增量规格解决什么

本文件更新 `2026-08-09-book-video-workstation-v0.1-design.md` 中已经变化的部分。未被本文件明确覆盖的安全、状态机、人工审核、IndexTTS2、字幕、FFmpeg 和最终质检规则继续有效。

本次更新确定四件事。

1. V0.1 同时接受完整电子书和“书名 + 作者”两种输入。
2. 运营者已声明与出版社直接合作并取得图书宣传权。程序不建立版权认证、证明上传、ISBN 审批或授权材料保存流程。
3. 文案由 Grok 与 Codex 分工完成，事实和整书价值先锁定，再单独调用本机 `human-writing` Skill 去 AI 味。
4. 首个视频提供方改为 Grok CLI 原生媒体工具；MiniMax H3 保留手工切换接口。

V0.1 仍然只做 CMD 工作流，不做网页、GUI、自动发布和复杂多 Agent 平台。

## 2. 成功结果

用户可以只执行：

```powershell
bv new --title "书名" --author "作者"
```

随后按 CMD 提示完成：

```text
联网研究和整书理解
→ 整本书给读者的价值主题
→ 45 至 60 秒口播文案
→ human-writing 自然化
→ 人工文案审核
→ IndexTTS2 完整旁白
→ 字幕和旁白时码
→ 4 个左右的 15 秒分镜
→ Grok CLI 参考图和分段视频
→ S01 样片审核
→ 其余片段生成
→ FFmpeg 成片
→ 最终人工审核
```

完整电子书仍可直接导入：

```powershell
bv new "E:\电子书\某本书.epub"
```

两种输入最终进入相同的价值卡、文案、分镜、视频和渲染流程。

## 3. 输入和书籍身份

### 3.1 完整电子书模式

支持 TXT、PDF 和 EPUB 单文件。原文件只复制到工作区，不覆盖、不移动。正文按原有解析、章节分析和证据绑定流程处理。

### 3.2 书名和作者模式

必填字段只有：

```text
title
author
```

程序自动建立规范化 `book_id`，并通过 Grok 联网研究补充可获得的出版社信息、目录、作者访谈、合法试读、可靠长评和读者讨论。ISBN、译者、版次和出版社可以作为内部元数据保存，但不成为用户审批关口，也不要求用户补齐。

只有两种情况允许停止：

1. 书名和作者仍无法对应到一部明确作品。
2. Grok 和 Codex 均未获得任何足以支撑非虚构文案的材料。

一般的来源缺失、版本信息不完整或没有合法全文不阻断流程。系统继续使用可核验的公开材料完成最佳可负责的整书价值解释，不假装拥有没有取得的完整原文。

### 3.3 宣传权边界

系统接受运营者已经取得出版社宣传权这一业务前提，不验证、不保存证明，也不重复询问。该前提允许系统为账号制作图书推荐文案和视频，不改变以下内容准确性规则：

- 不伪造作者原话。
- 不把读者评论写成书中原文。
- 不把模型推测写成已证实事实。
- 真实书封由用户或出版社素材提供，生成模型不重画正式商品封面。

## 4. 整书理解和文案分工

### 4.1 E001 的主题

第一条视频固定回答：

> 这本书看见了读者生活中的什么问题，它如何帮助读者重新理解自己、他人或现实，它最终给读者带来什么价值。

E001 不从书中随机抽一个观点扩写，也不做章节摘要、作者生平或知识点罗列。允许用两到三个互相关联的概念支撑整书价值，但这些概念必须组成一条完整解释。

### 4.2 Grok 第一轮研究

Grok 使用客户端订阅和当前登录态完成联网研究，输出受 JSON Schema 约束的 `book_research.json`。至少包含：

- 书籍身份和基本背景。
- 全书试图回答的核心问题。
- 主要论证链或叙事推进。
- 两到五个相互关联的关键思想。
- 这些思想对应的现实生活处境。
- 读者可能获得的理解、选择或行动变化。
- 常见误读、争议和适用边界。
- 支撑重要判断的来源和材料身份。
- Grok 不确定或无法核实的内容。

Grok 不直接写最终口播，也不决定事实是否通过。

### 4.3 Codex 综合和证据锁

Codex 读取电子书解析产物或 `book_research.json`，生成：

```text
book_value_card.json
fact_lock.json
script_draft.md
```

`book_value_card.json` 负责整本书的价值结构。`fact_lock.json` 保存不能在后续自然化中改变的书名、作者、概念、事实、限定词、否定词、已核实引语和禁止归因项。

Codex 第一稿必须把书和人的生活连接起来。每个抽象概念至少对应一个可观察的生活处境、选择或关系变化。现实场景只用于帮助读者理解，不伪造成真实读者证言，也不写成作者亲历。

### 4.4 Grok 第二轮挑刺

Grok 审查 `book_value_card.json` 和第一稿，重点报告：

- 是否把整本书缩成一个孤立观点。
- 是否误解作者的核心问题或论证关系。
- 是否缺少真实生活联系。
- 是否存在空洞鸡汤、强行带货或模板化 AI 句式。
- 是否把二手观点错写成作者观点。
- 哪些句子在视觉上无法表现。

Grok 输出结构化审查，不直接覆盖文案。Codex 逐项接受、拒绝或改写，并记录决策。

## 5. Human Writing 去 AI 味

### 5.1 固定位置

`human-writing` 只在事实锁和整书价值结构通过以后调用，位于最终文案审核之前、分镜之前：

```text
研究和整书理解
→ fact_lock
→ 文案第一稿
→ Grok 挑刺
→ human-writing 自然化
→ 事实回归检查
→ 人工文案审核
```

它不能参与书籍身份判断，不能增加未经来源支撑的事实，也不能为了口语化修改书名、作者、概念含义和限定条件。

### 5.2 调用材料

Codex 调用本机 Skill：

```text
%USERPROFILE%\.codex\skills\human-writing\SKILL.md
```

口播任务同时遵循该 Skill 的：

```text
references/reality.md
references/formats.md
```

初稿完成后再读取：

```text
references/revision.md
```

随后运行：

```powershell
python %USERPROFILE%\.codex\skills\human-writing\scripts\check_prose.py <script-path>
```

Skill 文件和直接引用规则文件的 SHA-256 写入阶段清单。Skill 改变后，自然化阶段及其下游标记为 stale。

### 5.3 口播验收

最终文案必须：

- 可以直接念，不使用教程、报告和宣讲口气。
- 第一句尽快进入一个人与生活有关的问题。
- 顺着听众自然会问的下一句推进。
- 不虚构用户经历、读者证言、作者原话或现实细节。
- 不靠空泛哲理和同义反复填满 45 至 60 秒。
- 通过 `check_prose.py` 的硬禁令检查。
- 重新通过 `fact_lock` 确定性检查和 Codex 语义检查。
- 使用 IndexTTS2 真实生成后落在 45 至 60 秒；字符数只作写稿参考，不作为最终时长证明。

## 6. Grok CLI 视频提供方

### 6.1 已验证能力

本机 Grok CLI 1.0.4 已使用 `grok.com` 登录。当前代理会话原生暴露：

```text
image_gen
image_edit
image_to_video
reference_to_video
```

没有独立的原生 `text_to_video` 工具。V0.1 因此采用“先生成参考图，再生成视频”的稳定路径。

`reference_to_video` 支持本地绝对路径参考图、9:16、480p/720p 和最长 15 秒，并返回已保存视频的本地绝对路径。工具暴露和参数已经核验；订阅额度下的一次真实媒体生成仍属于首个私有样片验收项。

### 6.2 提供方接口

新增中性 `VideoProvider` 边界，第一版配置允许：

```yaml
video_generation:
  provider: grok_cli
```

可切换值：

```text
grok_cli
grok_manual
h3_manual
```

- `grok_cli` 是默认自动路径。
- `grok_manual` 在原生工具失败时允许用户从 Grok 客户端手工生成并导入。
- `h3_manual` 保留 MiniMax H3 手工生成和导入能力。

V0.1 不实现 xAI 付费 API 和 H3 API。现有 H3 导入、哈希、FFprobe 和原子发布代码作为底层兼容实现复用；外部命令、状态和新清单改用中性 video/provider 语义。

每个片段清单保存实际提供方。更换提供方会使受影响的提示词、参考图、片段、渲染和 QC 失效，不能静默混用来源不明的旧素材。

### 6.3 参考图和连续性

每个 E001 生成：

```text
character_reference.png
style_reference.png
scene_S01_keyframe.png
scene_S02_keyframe.png
scene_S03_keyframe.png
scene_S04_keyframe.png
```

所有片段复用人物和风格参考图，并加入当前场景关键帧。参考图、连续性圣经、分镜和提示词均绑定批准文案、旁白时码和语义锁哈希。

真实书封不交给模型重画，只在 FFmpeg 包装阶段叠加。

### 6.4 分段生成

完整旁白仍是唯一主时间轴。通常生成 S01 至 S04，每段不超过 15 秒；必要时允许 S05。

测试顺序固定为：

1. 只生成 S01。
2. 下载到隔离接收目录。
3. 复制、哈希和 FFprobe 校验。
4. 标记 `visual_sample_ready`，等待人工查看。
5. S01 通过后才逐段生成剩余视频。

每个片段默认无人物对白。Grok 原生环境声可以经检查后低音量保留，IndexTTS2 完整旁白始终覆盖全片。

### 6.5 Grok CLI 运行限制

运行时视频会话只允许媒体生成所需工具，不授予终端、源码编辑或任意文件写入能力。输入只包含当前片段需要的提示词和参考图，不包含完整电子书、其他项目文件、环境变量、Cookie 或 Token。

视频生成属于外部额度操作，命令必须显式带：

```powershell
--allow-external
```

每次命令只生成一个 shot_id。S01 未通过时不自动生成 S02 至 S04。

## 7. 更新后的 CMD 流程

### 7.1 只有书名和作者

```powershell
bv new --title "书名" --author "作者"
bv next <book-id> E001 --allow-external
bv open <book-id> E001 script
bv approve <book-id> E001 script
bv next <book-id> E001 --allow-external
bv generate-video <book-id> E001 S01 --allow-external
bv open <book-id> E001 video
bv approve <book-id> E001 visual-sample
bv generate-video <book-id> E001 S02 --allow-external
bv generate-video <book-id> E001 S03 --allow-external
bv generate-video <book-id> E001 S04 --allow-external
bv next <book-id> E001
bv open <book-id> E001 final
bv approve <book-id> E001 final
```

如果自动视频失败，可以执行：

```powershell
bv import-video <book-id> E001 S01 "D:\Downloads\S01.mp4"
```

### 7.2 完整电子书

电子书模式只替换第一条命令，其余一致：

```powershell
bv new "E:\电子书\某本书.epub"
```

## 8. Grok CLI 开发委派

Sol 负责规格、任务分解、写入范围、权限、审查、测试、返工和最终验收。主要实现和对应测试交给 Grok CLI。

Grok CLI 开发任务至少拆成：

1. 书名和作者输入及研究产物。
2. Codex/Grok 内容运行时适配和 `human-writing` 阶段。
3. 中性视频提供方和现有 H3 兼容迁移。
4. Grok CLI 参考图和视频生成适配器。
5. Task 21B 真实运行时装配和外部调用闸门。
6. CMD、文档、集成测试和失败恢复。

每项由 Grok CLI 在限定文件中实现并运行针对性测试。Sol 独立检查 diff、错误码、状态失效、路径边界、敏感信息、外部调用、额度控制和全量测试。重大返工继续交给 Grok CLI，小范围安全加固可由 Sol 完成并明确记录。

## 9. 错误和恢复

新增或更新稳定错误码：

```text
book_identity_unresolved
book_research_failed
book_research_empty
human_writing_unavailable
human_writing_check_failed
fact_lock_mismatch
video_provider_not_configured
video_generation_not_authorized
video_generation_failed
video_output_missing
video_output_unsafe
visual_sample_not_approved
```

模型和媒体工具的原始输出、绝对私有路径、登录信息和服务端错误体不得进入公开错误码。失败产物保留在受控工作区，`bv retry` 只重跑失败或 stale 阶段。

## 10. 第一版验收

### 10.1 自动化测试

- 书名和作者可以创建唯一项目和 E001。
- 电子书模式保持兼容。
- 没有 `--allow-external` 时不调用 Grok、Codex、火山或视频工具。
- Grok 研究输出必须通过 Schema，空输出不能完成阶段。
- `human-writing` 缺失、哈希变化、检查失败和事实锁变化均能正确阻断或失效下游。
- 视频提供方切换不会混用旧片段。
- S01 未批准时不能自动生成其余片段。
- Grok 返回的路径经过工作区、文件类型、哈希和 FFprobe 校验。
- H3 手工导入的现有安全行为保持不变。
- 现有测试和新增测试全部通过。

### 10.2 私有真实验收

用户提供第一本书的书名和作者以后，依次验证：

1. Grok 能完成真实联网研究。
2. Codex 能生成整书价值卡、事实锁和第一稿。
3. Grok 挑刺被逐项处理。
4. `human-writing` 成稿通过检查且听起来可以直接念。
5. IndexTTS2 真实旁白在 45 至 60 秒。
6. Grok CLI 使用订阅额度生成一条 15 秒、9:16 的 S01。
7. S01 通过后再完成其他片段。
8. FFmpeg 输出带完整旁白、字幕和真实书封的最终 MP4。
9. 技术 QC 和人工成片审核均通过。

在第 6 项完成前，只能声称 Grok CLI 视频工具已发现和已接线，不能声称订阅视频生成已经跑通。在第 8 项完成前，不能声称整条私有流水线已经验收完成。

## 11. 明确不做

- 不做版权认证、授权证明上传、ISBN 审批或出版社合同管理。
- 不自动寻找或下载盗版电子书全文。
- 不为了书名输入建立复杂知识库、向量库或数据库。
- 不让多个 Agent 自由对话或自行决定流程。
- 不使用 Grok 各片段原生对白拼成完整旁白。
- 不自动发布到抖音。
- 不做 GUI、网页工作台、账号矩阵或运营数据看板。
- 不实现 xAI 视频 API 或 H3 API。

## 12. 覆盖关系

本规格覆盖旧设计中的以下决定：

- “输入必须是本地完整电子书”改为“完整电子书或书名 + 作者”。
- “Grok 只做非阻塞审查”改为“Grok 负责研究和挑刺，Codex 负责证据综合和最终决策”。
- “H3 手工导入是唯一视频路径”改为“Grok CLI 是默认自动路径，Grok/H3 手工导入是回退和切换路径”。
- 旧的 H3 专用外部命令和新清单改用中性视频提供方语义；底层安全导入实现保留兼容。

旧设计关于 E001 整本书价值、后续主题去重、完整旁白主时间轴、人工文案与成片审核、敏感信息保护、原子写入、哈希、断点续跑和 fail-closed 状态机的要求继续有效。
