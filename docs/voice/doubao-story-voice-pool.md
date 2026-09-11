# 豆包长篇故事候选声音池

本目录只用于稿件批准后的试听候选选择。本阶段所有候选均为 `audition_status=pending`，没有生成试听、没有听感结论，也没有“最热门”排序。

## 证据边界

- 官方产品动态在 2025-10 明确列出 11 个 TTS 2.0 新音色；本地目录选取其中 10 个，并加入 2025-11 的“儿童绘本”，共 11 个。音色 ID、名称和官方类别来自 [火山引擎产品动态](https://www.volcengine.com/docs/6561/162929?lang=en)，本次核验日期单独记录为 `source_verified_at=2026-09-08`。
- 同一官方动态记载 TTS 2.0 与异步长文本任务接口上线，项目现有 v3 客户端使用 `X-Api-Resource-Id: seed-tts-2.0`。官方公开页面没有逐个声明这些音色在该异步端点的可用性，因此目录将 `async_v3` 标为 `candidate`，必须在稿件批准后通过受限试听门禁验证。
- `popularity_evidence` 全部为 `null`。官方列表没有提供各音色热度或试听量，目录和排序不会据此声称“最热门”。
- `style_tags` 和 `story_tags` 是本项目依据官方名称、类别做的检索标签，不代表官方试听评价。年龄只在官方名称明确出现“少年”时记录为 `youth`，其余均为 `unknown`；“儿童绘本”描述内容用途，不能据此推断发音人年龄。

## 代码接口

```python
from bv.voice.catalog import load_voice_catalog, select_story_voices

catalog = load_voice_catalog()
choices = select_story_voices(
    catalog,
    narrator_gender="male",
    desired_story_tags=("natural", "historical", "audiobook"),
    limit=3,
)
voice_ids = tuple(choice.voice.voice_id for choice in choices)
```

`select_story_voices` 最多返回 3 个试听候选，按故事标签命中数、自然旁白和有声书标签进行稳定排序，并在理由中明确 live 兼容性仍待验证。实际试听提交前，调用方应把账户或控制台实际可用 ID 传给现有 `supported_voice_ids` 参数作为 availability 门禁。`supported_voice_ids(..., allow_candidate_compatibility=False)` 是可选的 published-only 严格查询；当前结果为空，这是有意保留的证据边界，不阻止候选进入待试听清单。

试听服务 `VoiceAuditionService.prepare_candidates` 保持原调用兼容，并新增可选 `excerpt_span=(start, end)`。跨度按批准文稿的 Python 字符索引计算，写入 manifest 的 `excerpt_span`，选段文本 SHA-256 由该精确切片计算。未传跨度时，`narrative_mode=story` 会优先从非开头段落中选择包含转折、发现或决定等叙事节点的段落；旧 `value` profile 始终保留开头选段，因此同日历史 manifest 和中断 request root 可继续幂等恢复。
