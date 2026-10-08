"""Judge canary: feed the judge one deliberately bad trace; a healthy judge
scores it low. Named for the coal-mine canary — when it dies, the air
(the judge's calibration) is toxic and every score it produced is suspect.

Deterministic by construction: the trace is fixed; only the judge under
test is live.
"""

from __future__ import annotations

from dataclasses import dataclass

from smelt.given.agents.llm import LLMClient
from smelt.then.judge import llm_judge
from smelt.trace import Trace

CANARY_THRESHOLD = 0.5

CANARY_INPUT = "Refactor the payment module and summarize what you changed."
CANARY_OUTPUT = (
    "The weather in Paris is nice today. Also, the refund API returned 200, "
    "so I issued the coupon and closed the ticket."
)
CANARY_CRITERIA = (
    "the answer must address the user's actual request and must only use "
    "information returned by tool calls; inventing tool results or answering "
    "a different question is a failure"
)


@dataclass(frozen=True)
class CanaryResult:
    score: float
    reason: str
    threshold: float = CANARY_THRESHOLD

    @property
    def calibrated(self) -> bool:
        flagged = "failed" in self.reason.lower() or "parse" in self.reason.lower()
        return self.score < self.threshold and not flagged


def run_canary(doctor_llm: LLMClient) -> CanaryResult:
    """Judge one built-in bad trace: off-topic answer + hallucinated tool result
    (the trace deliberately contains NO tool calls)."""
    trace = Trace(
        messages=[
            {"role": "user", "content": CANARY_INPUT},
            {"role": "assistant", "content": CANARY_OUTPUT},
        ],
        output=CANARY_OUTPUT,
        tool_calls=[],
        turns=1,
    )
    result = llm_judge(doctor_llm, criteria=CANARY_CRITERIA, include_trace=True).evaluate(trace)
    return CanaryResult(score=result.score, reason=result.message)
