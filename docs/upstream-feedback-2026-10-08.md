# Smelt upstream feedback — 2026-10-08

> Status 2026-10-08: all three items are fixed in this repo (see CHANGELOG
> "Unreleased"), with regression tests in `tests/test_lint_false_positives.py`.

Three issues verified against smelt @ 2b57035 (v0.4.1). Items 1 and 2 are new
false positives reported by downstream skill authors; item 3 was raised before
and is still open.

## 1. assets: `PROSE_PATH_RE` re-matches markdown link targets with a trailing `)`

**Symptom.** Any skill whose body links to a local file with markdown syntax,
e.g. `[the guide](references/guide.md)`, gets a phantom
`referenced file does not exist: references/guide.md)` error — note the trailing
`)`. Every bilingual skill using markdown links is affected (downstream examples:
tech-documentation / svelte / git-guru, whose assets scores sit at 40–60 purely
because of this).

**Root cause.** `src/smelt/lint/checks/assets.py:15`:

```python
PROSE_PATH_RE = re.compile(r"(?<![\w/`])((?:scripts|templates|assets|references|examples|docs|tests)/[^\s`\"'\]]+)")
```

The negative lookbehind only excludes word chars, backtick and `/`, so a path
preceded by `(` (i.e. a markdown link target) still matches. The character class
`[^\s`\"'\]]+` does not exclude `)`, so the match swallows the link's closing
paren. `rstrip(".,;:!?")` on line 37 does not strip `)`. `LOCAL_LINK_RE` captures
the correct path, but the phantom `...)` variant lands in the same `refs` set and
fails the existence check.

**Minimal repro.**

```python
body = "See [the guide](references/guide.md) for details."
LOCAL_LINK_RE.findall(body)                    # ['references/guide.md']   ok
[m.group(1).rstrip(".,;:!?")
 for m in PROSE_PATH_RE.finditer(body)]        # ['references/guide.md)']  phantom
```

With 20 points deducted per missing ref (capped at 60), a few markdown links
push the assets score to 40–60 even though every referenced file exists.

**Suggested fix.** Exclude `)` (and probably `(`, `<`, `>`) from the character
class, or add `)` to the `rstrip` set; more robustly, skip prose matches whose
span is already covered by `LOCAL_LINK_RE`.

## 2. clarity: `\{\{.*?\}\}` scans fenced code blocks, flags Vue interpolation

**Symptom.** A skill documenting Vue template syntax gets
`found unfilled content: template placeholder {{...}}` and loses 15 points.
Downstream example: zh-cn/frontend-vue clarity 85, and this is the only
deduction.

**Root cause.** `src/smelt/lint/checks/clarity.py:11` searches `skill.body`
directly:

```python
(re.compile(r"\{\{.*?\}\}"), "template placeholder {{...}}"),
```

There is no code-block stripping anywhere in the check, so `{{ message }}`
inside a fenced ```` ```vue ```` block is indistinguishable from a genuine
unfilled placeholder in prose.

**Minimal repro.**

```python
body = "Use interpolation:\n\n```vue\n<p>{{ message }}</p>\n```\n"
re.compile(r"\{\{.*?\}\}").findall(body)       # ['{{ message }}']  false positive
```

**Suggested fix.** Strip fenced code blocks (and ideally inline code spans)
from the body before applying the placeholder patterns, at least for the
`{{...}}` rule. TODO/FIXME scanning may want to keep code blocks, so consider
per-pattern flags.

## 3. `__version__` out of sync with `pyproject.toml` (still open)

`src/smelt/__init__.py:22` has `__version__ = "0.4.0"` while
`pyproject.toml:7` has `version = "0.4.1"` at the v0.4.1 release commit
(2b57035). Anything reading `smelt.__version__` reports the wrong version.

**Suggested fix.** Single-source the version, e.g.
`__version__ = importlib.metadata.version("smelt")`, or add a release-checklist
step / CI assertion that the two match.
