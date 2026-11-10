from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason
