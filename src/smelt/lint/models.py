"""Data model definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    """Issue severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @classmethod
    def max(cls, severities: list[Severity]) -> Severity:
        order = {cls.INFO: 0, cls.WARNING: 1, cls.ERROR: 2}
        if not severities:
            return cls.INFO
        return max(severities, key=lambda s: order[s])


@dataclass
class Message:
    severity: Severity
    text: str

    def __str__(self) -> str:  # pragma: no cover - display helper
        icon = {Severity.INFO: "ℹ", Severity.WARNING: "⚠", Severity.ERROR: "✖"}[self.severity]
        return f"{icon} {self.text}"


@dataclass
class CheckResult:
    """Result of a single check."""

    check_id: str
    name: str
    score: float  # 0..100
    passed: bool
    messages: list[Message] = field(default_factory=list)

    @property
    def severity(self) -> Severity:
        return Severity.max([m.severity for m in self.messages])


@dataclass
class SkillDoc:
    """A parsed SKILL.md."""

    path: Path  # path to the SKILL.md file
    name: str  # name from frontmatter (directory name when missing)
    description: str
    frontmatter: dict
    body: str  # body without frontmatter
    sections: list[tuple[int, str]] = field(default_factory=list)  # [(heading level, title)]
    word_count: int = 0

    @property
    def dir(self) -> Path:
        return self.path.parent


@dataclass
class SkillReport:
    """Full lint report for one skill."""

    skill_path: Path
    results: list[CheckResult]
    total_score: float
    grade: str

    @property
    def passed(self) -> bool:
        return self.grade != "F"
