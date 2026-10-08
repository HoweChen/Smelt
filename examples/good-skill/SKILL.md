---
name: good-skill
description: Use this skill when the user asks for a weekly report or wants scattered work notes organized; it compiles raw records into a structured weekly report document.
---

# Good Skill: Weekly Report Example

## Overview

This skill turns scattered work records into a structured weekly report. The
output is organized into four sections — Done this week / In progress / Risks &
blockers / Next week — so it can be pasted straight into a team reporting system.

## When to use

- When the user says "write my weekly report" or "organize this week's progress",
  use this skill.
- When the user provides raw work notes and wants a structured document, use
  this skill.
- Common trigger words: weekly report, progress summary, weekly summary.

## Steps

1. Collect the user's raw work records (chat logs, ticket lists, anything).
2. Classify items into the four sections and list them as bullet points.
3. Run `scripts/hello.py` to generate a draft report (a placeholder script for
   demonstration).
4. Confirm sectioning and wording with the user before producing the final
   document.

## Example

Input: "This week I finished the login module; the integration is blocked on the
third-party payment API."

Output: a weekly report in four sections, where "Risks & blockers" notes
"third-party payment integration blocked, expected to be resolved next week".

## Caveats

- All numbers and facts must come from user-provided material; never fabricate
  progress.
- If there are too few records to draft a report, ask the user for more material
  first.
- Keep the tone factual and neutral; avoid subjective evaluation.
- When a deadline is mentioned, quote it verbatim instead of paraphrasing.
- Do not merge unrelated items into a single bullet; one bullet per fact.
- If two items conflict (for example two different dates for the same task),
  surface the conflict explicitly and ask which one is correct before drafting.
- The generated draft is a starting point, not the final word: always offer the
  user a chance to adjust section membership and wording before finishing.
