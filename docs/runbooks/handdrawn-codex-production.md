# Codex 对话内图书手绘视频生产手册

> 本文保留旧短视频/value 模式的生产说明。新的长篇故事创作以 [长篇故事生产手册](longform-story-production.md) 和项目 `AGENTS.md` 为准；不沿用下文的短时长、固定图片数或 human-writing 前置要求。

## 当前边界

这套流程在 Codex 对话中编排，不新增网页或 GUI，也不自动上传内容平台。CLI 负责项目、状态和门禁等基础动作，手绘成片后半程由 Agent 按合同调用本地模块和已授权供应商。

新建标题作者 Episode 使用短视频生产配置。口播硬范围 30–45 秒，目标 35–40 秒，总计四张背景图，固定第 11 种 `retro-gouache-concept` 画风。缺少 `production_profile.json` 的旧 Episode 继续使用 45–60 秒兼容配置。

## 输入

每本书接受两种输入：

- 书名加作者
- 无打开限制且有权使用的 TXT、PDF、EPUB 单文件

标题作者模式应先建立书籍身份和证据边界。安装微信读书 Skill 时，可在用户有权访问的范围内检查元数据、目录、划线、笔记和书评。没有合法来源时应停下请求材料，不能把模型记忆当成整本书证据。

## 真实生产准备

- Python 3.11 与 uv
- FFmpeg、FFprobe、Node.js、npm
- 选定的 TTS 路线与合法音色授权
- 火山 ASR 所需环境变量
- 与实际商品版本一致的真实书封
- 可渲染中文的字幕字体

凭据只能放在本机环境变量或私密凭据存储中。对话和公开日志只报告 `SET` 或 `UNSET`。

## 固定生产顺序

### 1. 文稿门禁

Agent 先完成书籍身份、证据边界、整书核心矛盾、目标读者和现实适用边界，再生成读者价值口播稿。新短视频稿件必须能在 30–45 秒内自然朗读。

文稿批准前，不调用 TTS、ASR、图像生成或视频渲染。

### 2. 音色试听门禁

用户批准文稿后，从批准稿截取同一个连续片段，为最多三个候选音色生成 15–25 秒试听。一次授权只覆盖已说明的文稿哈希和音色 ID。

用户确认音色以后才能生成正式旁白。失败时不自动重试，不换音色，也不回退到别的供应商。

### 3. 正式音频与字幕时间轴

正式旁白统一为完整 48 kHz 单声道 WAV。火山 ASR 只对这份最终 WAV 做字级时间对齐。最终音频时长是字幕、场景和渲染的唯一全局时钟。

原始 TTS 响应、任务回执、签名 URL 和试听音频留在 Episode `.private` 目录。

### 4. 四场景与代表图门禁

新短视频固定四个背景场景，画风固定为第 11 种 `retro-gouache-concept`。先生成开头、中段、结尾三张代表图。

代表图需要检查人物年龄、脸、发型、服装、关键物件、场景逻辑、画内文字、水印、假书封、手部和肢体。批准记录绑定三张图、分镜、画风和人物锁的 SHA-256。

代表图批准后，只补第四张背景图。任何被绑定的内容变化都会使批准失效。

### 5. 静音画面与最终合成

内置 Remotion 渲染器输出 1080×1920、30fps、H.264 的静音画面，四张背景之间使用 15 帧交叉溶解。FFmpeg 烧录 ASS 字幕，加入最终旁白和真实书封。

最终 MP4 使用 H.264/yuv420p 视频与 AAC 48 kHz 双声道音频，无旋转元数据，音画时长误差不超过一帧。

### 6. 独立作品封面

独立封面使用 1080×1440 的 3:4 规格。封面提示词来自已批准文稿的核心意义，生成调用需单独授权，返回图片需单独验收。封面不能代替视频中的真实书封。

### 7. 最终人工门禁

自动 QC 只确认文件、编码、分辨率、帧率、音轨、时长、清单和哈希。审美、人物一致性、手部瑕疵、情绪和叙事效果必须由人完整观看后确认。

明确批准视频和独立封面后，系统写入不可变批准记录，绑定候选清单、视频和封面哈希。沉默不能视为批准。

## 主要产物

```text
workspace/books/<book_id>/episodes/E001/
├─ production_profile.json
├─ script/approved.txt
├─ script/script_package.json
├─ cover/manifest.json
├─ media/voice/voice_master.wav
├─ media/alignment/alignment.json
├─ media/subtitles/subtitles.ass
├─ media/illustration/style_decision.json
├─ media/illustration/character_lock.json
├─ media/illustration/illustration_storyboard.json
├─ media/illustration/illustration_manifest.json
├─ media/render/picture_silent.mp4
├─ media/final/final.mp4
├─ media/final/social_cover.png
├─ media/final/final.render.json
└─ media/final/final.qc.json
```

私有原始请求和回执位于 Episode 的 `.private`。公开仓库和公开交接不得包含这些文件。

## 对话检查点

1. 用户提供书名作者或合法文件
2. Agent 展示整书价值文稿
3. 用户批准文稿
4. Agent 制作同片段音色试听
5. 用户确认音色
6. Agent 生成正式旁白、ASR、字幕、四场景分镜和三张代表图
7. 用户批准代表图
8. Agent 补齐第四张背景图并渲染视频
9. Agent 单独生成并展示 3:4 封面
10. 用户完整观看并最终确认

## 验收标准

技术验收：

- 时长 30–45 秒，或符合旧 Episode 的已批准生产配置
- 1080×1920、30fps、H.264/yuv420p
- AAC、48 kHz、双声道
- 总计四张背景图，第 11 种固定画风
- 字幕位于安全区，音画误差不超过一帧
- 独立封面 1080×1440
- 渲染清单、QC 报告和输入哈希可复核

人工验收：

- 文稿准确覆盖整本书的读者价值和现实边界
- 声音自然，没有明显做作感和错误发音
- 人物与关键物件在四个场景保持一致
- 没有黑边、乱码、假书封、Logo、水印和明显肢体问题
- 文稿、画面、字幕和情绪属于同一条叙事

公开仓库中的自动测试覆盖假外部提供方、真实 Remotion 与真实 FFmpeg 路径。真实云端账号、额度、音色授权和审美结果仍需按 Episode 单独验收，私有媒体证据不随代码发布。
