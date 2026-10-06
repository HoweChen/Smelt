"""SmeltCase: fluent, immutable given / when / then case definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from smelt.given.context import ContextSpec
from smelt.then.expectations import Expectation
from smelt.when.inputs import CaseInput

if TYPE_CHECKING:
    from smelt.given.agents.base import Agent
    from smelt.results import CaseResult


@dataclass(frozen=True)
class SmeltCase:
    """A single skill-verification case.

    Every ``given`` / ``when`` / ``then`` returns a new instance, so definitions
    are safe to reuse and derive (e.g. one agent with several whens)::

        base = new_case("commit").given(smelt_agent(...))
        a = base.when(text("commit this")).then(tool_call("git"))
        b = base.when(text("don't touch my code")).then(no_tool_call("git"))
    """

    name: str = "unnamed"
    contexts: tuple[ContextSpec, ...] = ()
    agent: Agent | None = None
    fragments: tuple = ()  # skill()/llm()/tools() segments, assembled at run time
    trigger: CaseInput | None = None
    expectations: tuple[Expectation, ...] = ()
    keep_workspace: bool = False  # True keeps the workspace under .smelt/ for debugging

    # -- given ---------------------------------------------------------------
    def given(self, item) -> SmeltCase:
        """Stack a context, attach a whole agent, or add an agent fragment (skill/llm/tools)."""
        from smelt.given.agents.base import Agent as AgentProtocol
        from smelt.given.fragments import LLMSpec, SkillSpec, ToolsSpec

        if isinstance(item, ContextSpec):
            return replace(self, contexts=self.contexts + (item,))
        if isinstance(item, (SkillSpec, LLMSpec, ToolsSpec)):
            return replace(self, fragments=self.fragments + (item,))
        if isinstance(item, AgentProtocol):
            if self.agent is not None:
                raise ValueError(f"case {self.name!r} already has an agent; only one agent per case")
            return replace(self, agent=item)
        raise TypeError(
            f"given() accepts context() / agent / skill() / llm() / tools() only, got: {type(item).__name__}"
        )

    # -- when ----------------------------------------------------------------
    def when(self, trigger: CaseInput) -> SmeltCase:
        """Set the trigger: text(...) or directory(...). One trigger per case."""
        if self.trigger is not None:
            raise ValueError(f"case {self.name!r} already has a trigger; only one trigger per case")
        return replace(self, trigger=trigger)

    # -- then ----------------------------------------------------------------
    def then(self, *expectations: Expectation) -> SmeltCase:
        """Append expectations; chainable."""
        for e in expectations:
            if not isinstance(e, Expectation):
                raise TypeError(
                    f"then() accepts expectation objects (tool_call / text_similar / ...) only, got: {type(e).__name__}"
                )
        return replace(self, expectations=self.expectations + tuple(expectations))

    # -- run -----------------------------------------------------------------
    def run(self) -> CaseResult:
        """Execute the case and return a CaseResult; use .assert_passed() under pytest."""
        from smelt.runner import run_case

        return run_case(self)

    # -- report --------------------------------------------------------------
    def report(self, *, output_dir: str | None = None, quiet: bool = False) -> CaseResult:
        """Run the case, write an HTML report under .smelt/reports/, return the result.

        The report path lands on ``result.report_path``; ``result.assert_passed()``
        still works for pytest.
        """
        from smelt.report import write_html_report

        result = self.run()
        path = write_html_report(result, output_dir or ".smelt/reports")
        result.report_path = str(path)
        if not quiet:
            print(f"report written to {path}")
        return result

    def report_cli(self, *, quiet: bool = False) -> CaseResult:
        """Run the case and print a formatted terminal report; return the result."""
        from smelt.report import render_text

        result = self.run()
        if not quiet:
            print(render_text(result))
        return result


def new_case(name: str = "unnamed", *, keep_workspace: bool = False) -> SmeltCase:
    """Start a new case definition."""
    return SmeltCase(name=name, keep_workspace=keep_workspace)
