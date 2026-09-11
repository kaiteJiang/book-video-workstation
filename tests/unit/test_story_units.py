import hashlib
from types import SimpleNamespace

import pytest

from bv.illustration.story_units import validate_story_units
from bv.workflow.media_stages import _story_scene_boundaries


def plan():
    text = '童年等待。争论持续多年。成年终于获立。'
    unit = dict(id='U01', start=0, end=len(text), beat='等待到获立',
                before=dict(time='童年', place='侧间', state='储位未定', image='蓝衣儿童'),
                events=['君臣争论多年'],
                after=dict(time='成年', place='正殿', state='获立太子', image='红衣青年'),
                change_kind='situation', state_change='储位确定', story_elapsed='多年',
                evidence=['核查材料'], b_entry_text='成年终于获立。', next_link='成长无法返还',
                semantic_review='正文包含多年争论，两端年龄与身份均发生变化')
    return text, dict(schema_version='narrative-unit-v1', source_sha256=hashlib.sha256(text.encode()).hexdigest(), units=[unit])


def test_complete_story_unit_passes():
    text, payload = plan()
    assert len(validate_story_units(payload, text)) == 1


@pytest.mark.parametrize('field,value,error', [
    ('events', [], 'ab_events_missing'),
    ('change_kind', 'gesture', 'ab_change_kind_invalid'),
    ('start', 1, 'ab_span_invalid'),
    ('b_entry_text', '不存在', 'ab_entry_text_invalid'),
    ('b_entry_text', '童年等待。', 'ab_entry_at_start'),
    ('semantic_review', '', 'ab_semantic_review_missing'),
])
def test_invalid_unit_is_blocked(field, value, error):
    text, payload = plan()
    payload['units'][0][field] = value
    with pytest.raises(ValueError, match=error):
        validate_story_units(payload, text)


def test_changed_script_and_unchanged_state_are_blocked():
    text, payload = plan()
    with pytest.raises(ValueError, match='ab_script_hash_mismatch'):
        validate_story_units(payload, text + '新增')
    payload['units'][0]['after']['state'] = payload['units'][0]['before']['state']
    with pytest.raises(ValueError, match='ab_state_unchanged'):
        validate_story_units(payload, text)


def test_strict_planning_never_splits_one_unit_to_fill_three_slots():
    with pytest.raises(ValueError, match='ab_units_need_replanning_not_mechanical_split'):
        _story_scene_boundaries([SimpleNamespace(start=0)], SimpleNamespace(characters=[]), (), 120000, strict=True)
