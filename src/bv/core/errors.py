from pathlib import Path


class BVError(Exception):
    def __init__(
        self,
        code: str,
        stage: str,
        user_message: str,
        next_command: str | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.code = code
        self.stage = stage
        self.user_message = user_message
        self.next_command = next_command
        self.log_path = Path(log_path) if log_path is not None else None
        super().__init__(user_message)
