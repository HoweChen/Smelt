"""Reference-testing assertions: args negation, reach, restraint, untouched, mock."""

from smelt import fixed_agent, new_case, no_tool_call, text


def test_no_tool_call_with_args_subset():
    agent = fixed_agent("done", tool_calls=[
        {"name": "read_file", "arguments": {"path": "references/endpoints.md"}},
    ])
    blocked = (
        new_case("restraint")
        .given(agent)
        .when(text("go"))
        .then(no_tool_call("read_file", args={"path": "references/endpoints.md"}))
        .run()
    )
    assert not blocked.passed
    assert "references/endpoints.md" in blocked.expectations[0].message

    other_file_ok = (
        new_case("restraint-ok")
        .given(agent)
        .when(text("go"))
        .then(no_tool_call("read_file", args={"path": "references/other.md"}))
        .run()
    )
    assert other_file_ok.passed


def test_no_tool_call_without_args_unchanged():
    agent = fixed_agent("done", tool_calls=[{"name": "read_file", "arguments": {"path": "x"}}])
    result = new_case("n").given(agent).when(text("go")).then(no_tool_call("read_file")).run()
    assert not result.passed  # name-only still blocks any call of that tool
