#!/usr/bin/env python3
from __future__ import annotations

import re
import atomize as core

WH_RE = re.compile(r"(?i)\b(?:what|which|who|whom|whose|where|when|why|how)\b|(?:什么|哪个|哪些|多少|如何|是否|为何|为什么|哪一个|哪一种)")
AUX_AT_CLAUSE_START_RE = re.compile(
    r"(?im)(?:^|(?<=[.!?。！？]))[ \t]*(?:is|are|was|were|do|does|did|can|could|will|would|should|must|may)\b"
)


def first_request_match(masked: str):
    candidates = [m for m in (
        core.first_task_match(masked),
        WH_RE.search(masked),
        AUX_AT_CLAUSE_START_RE.search(masked),
    ) if m is not None]
    return min(candidates, key=lambda m: m.start()) if candidates else None

core.first_request_match = first_request_match


def self_test() -> None:
    # This is one proof request; "are" is declarative and must not be treated as a question lead.
    text = r"Let $1<p<\infty$. Suppose $\{f_n\}$ are functions such that $f_n\ge0$. If $f_n$ converges weakly to $f$, prove that $f\ge0$."
    parts, reason = core.split_atomic_problem(text)
    assert len(parts) == 1, (reason, parts)

    # A genuine question followed by a second imperative is still split.
    text2 = "What is the minimum value, and prove that it is attained."
    parts2, reason2 = core.split_atomic_problem(text2)
    assert len(parts2) == 2 and reason2 == "conjoined_tasks", (reason2, parts2)

    # An auxiliary question at clause start is recognized.
    text3 = "Is the map injective, and determine its image."
    parts3, reason3 = core.split_atomic_problem(text3)
    assert len(parts3) == 2 and reason3 == "conjoined_tasks", (reason3, parts3)


if __name__ == "__main__":
    self_test()
    raise SystemExit(core.main())
