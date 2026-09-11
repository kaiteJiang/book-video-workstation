"""Fail-closed structural preflight for narrative-unit A/B plans."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class StoryUnit:
    start: int
    end: int
    beat: str
    data: dict


def validate_story_units(payload: dict, text: str) -> tuple[StoryUnit, ...]:
    def require(condition, code):
        if not condition:
            raise ValueError(code)

    def nonblank(value):
        return isinstance(value, str) and bool(value.strip())

    require(isinstance(payload, dict), 'ab_plan_invalid')
    require(payload.get('schema_version') == 'narrative-unit-v1', 'ab_semantics_missing')
    require(payload.get('source_sha256') == hashlib.sha256(text.encode('utf-8')).hexdigest(), 'ab_script_hash_mismatch')
    values = payload.get('units')
    require(isinstance(values, list) and 1 <= len(values) <= 48, 'ab_units_invalid')
    cursor, identifiers, result = 0, set(), []
    for item in values:
        require(isinstance(item, dict), 'ab_unit_invalid')
        require(nonblank(item.get('id')) and item['id'] not in identifiers, 'ab_unit_id_invalid')
        identifiers.add(item['id'])
        start, end = item.get('start'), item.get('end')
        require(type(start) is int and type(end) is int and start == cursor and start < end <= len(text), 'ab_span_invalid')
        cursor = end
        for key in ('beat', 'state_change', 'story_elapsed', 'b_entry_text', 'next_link', 'semantic_review'):
            require(nonblank(item.get(key)), 'ab_' + key + '_missing')
        for key in ('before', 'after'):
            value = item.get(key)
            require(isinstance(value, dict) and all(nonblank(value.get(k)) for k in ('time', 'place', 'state', 'image')), 'ab_' + key + '_missing')
        for key in ('events', 'evidence'):
            value = item.get(key)
            require(isinstance(value, list) and bool(value) and all(nonblank(v) for v in value), 'ab_' + key + '_missing')
        require(item.get('change_kind') in {'situation', 'relationship', 'goal', 'consequence'}, 'ab_change_kind_invalid')
        require(item['before']['state'].strip() != item['after']['state'].strip(), 'ab_state_unchanged')
        require(item['before']['image'].strip() != item['after']['image'].strip(), 'ab_image_unchanged')
        segment = text[start:end]
        require(segment.count(item['b_entry_text']) == 1, 'ab_entry_text_invalid')
        require(segment.index(item['b_entry_text']) > 0, 'ab_entry_at_start')
        result.append(StoryUnit(start, end, item['beat'], item))
    require(cursor == len(text), 'ab_coverage_incomplete')
    return tuple(result)


def load_story_units(path: Path, text: str) -> tuple[StoryUnit, ...]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise ValueError('ab_plan_missing_or_invalid') from None
    return validate_story_units(payload, text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    parser.add_argument('--script', type=Path, required=True)
    args = parser.parse_args()
    try:
        units = load_story_units(args.plan, args.script.read_text(encoding='utf-8'))
    except ValueError as exc:
        parser.exit(1, str(exc) + '\n')
    print(json.dumps({'structural_check': 'pass', 'units': len(units), 'semantic_quality': 'requires_content_review'}, ensure_ascii=False))
