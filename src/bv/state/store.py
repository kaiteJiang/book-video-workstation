from pathlib import Path

from bv.core.atomic import atomic_write_json

from .events import Event, append_event
from .models import BookState, EpisodeState


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def load_book(self, book_id: str) -> BookState:
        return BookState.model_validate_json(
            self._book_path(book_id).read_text(encoding="utf-8")
        )

    def save_book(self, state: BookState) -> None:
        atomic_write_json(self._book_path(state.book_id), state.model_dump(mode="json"))

    def load_episode(self, book_id: str, episode_id: str) -> EpisodeState:
        return EpisodeState.model_validate_json(
            self._episode_path(book_id, episode_id).read_text(encoding="utf-8")
        )

    def save_episode(self, state: EpisodeState) -> None:
        atomic_write_json(
            self._episode_path(state.book_id, state.episode_id),
            state.model_dump(mode="json"),
        )

    def append_event(self, event: Event) -> None:
        append_event(self._events_path(event.book_id), event)

    def _book_path(self, book_id: str) -> Path:
        return self.root / "books" / book_id / "book.json"

    def _episode_path(self, book_id: str, episode_id: str) -> Path:
        return self.root / "books" / book_id / "episodes" / episode_id / "episode.json"

    def _events_path(self, book_id: str) -> Path:
        return self.root / "books" / book_id / "ledger" / "events.jsonl"
