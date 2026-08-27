from pathlib import Path
from typing import Any


class ScriptedStructuredModel:
    def __init__(self, responses: list[dict[str, Any]], fail_on_call: int | None = None):
        self.responses = list(responses)
        self.fail_on_call = fail_on_call
        self.calls = 0

    def complete(self, prompt: str, schema_type: type, request_dir: Path):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("scripted model failure")
        payload = self.responses.pop(0)
        return schema_type.model_validate(payload)


class FakeStage:
    def __init__(self, name: str, next_status: str, should_fail: bool = False):
        self.name = name
        self.next_status = next_status
        self.should_fail = should_fail
        self.calls = 0

    def run(self, context):
        self.calls += 1
        if self.should_fail:
            raise RuntimeError(f"{self.name} failed")
        return {"status": self.next_status}


class FakeVideoGateway:
    """In-memory video gateway for workflow tests; it never probes, copies, or calls providers."""

    def __init__(self, required_shot_ids: list[str]):
        self.required_shot_ids = tuple(required_shot_ids)
        self.present_shot_ids: list[str] = []

    def generate(self, book_id, episode_id, shot_id):
        raise RuntimeError("fake_video_gateway_generate_disabled")

    def import_video(self, book_id, episode_id, shot_id, source):
        if shot_id not in self.required_shot_ids:
            raise ValueError("invalid_segment_id")
        if shot_id not in self.present_shot_ids:
            self.present_shot_ids.append(shot_id)
        return self.status(book_id, episode_id)

    def status(self, book_id, episode_id):
        present = tuple(
            shot for shot in self.required_shot_ids if shot in self.present_shot_ids
        )
        missing = tuple(
            shot for shot in self.required_shot_ids if shot not in self.present_shot_ids
        )
        if not missing:
            workflow_status = "video_imported"
        elif present == ("S01",):
            workflow_status = "visual_sample_ready"
        elif present:
            workflow_status = "video_partial"
        else:
            workflow_status = "awaiting_video_generation"
        return {
            "status": workflow_status,
            "present_shot_ids": present,
            "missing_shot_ids": missing,
        }


FakeH3Gateway = FakeVideoGateway


class ArtifactStage(FakeStage):
    """A fake stage whose output can be removed to exercise manifest freshness."""

    def run(self, context):
        result = super().run(context)
        output = context.episode_root / ".test-stage" / self.name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(str(self.calls), encoding="utf-8")
        result["outputs"] = {self.name: output}
        return result
