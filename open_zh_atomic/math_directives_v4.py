#!/usr/bin/env python3
from __future__ import annotations

import re

# Only rewrite mathematical imperatives when they start a sentence/clause.  This avoids
# changing ordinary prose uses such as "the figure shows ..." while eliminating generic
# MT wording such as "查找" for contest-style "Find".
RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Let\b"), r"\1设"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Suppose\b"), r"\1假设"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Assume\b"), r"\1假设"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Given\b"), r"\1已知"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Find\b"), r"\1求"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Determine\b"), r"\1确定"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Compute\b"), r"\1计算"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Calculate\b"), r"\1计算"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Evaluate\b"), r"\1求值"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Prove(?:\s+that)?\b"), r"\1证明"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Show(?:\s+that)?\b"), r"\1证明"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Verify\b"), r"\1验证"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Solve\b"), r"\1求解"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Construct\b"), r"\1构造"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Describe\b"), r"\1描述"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Classify\b"), r"\1分类"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Explain\b"), r"\1解释"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Write\b"), r"\1写出"),
    (re.compile(r"(?im)(^|(?<=[.!?;:。！？；：])\s*)Express\b"), r"\1表示"),
)


def inject_directives(text: str) -> str:
    out = text
    for pattern, replacement in RULES:
        out = pattern.sub(replacement, out)
    return out


if __name__ == "__main__":
    sample = "Let x be real. Find the minimum. Show that it is unique. The graph shows symmetry."
    out = inject_directives(sample)
    assert out.startswith("设 x")
    assert "求 the minimum" in out
    assert "证明 it is unique" in out
    assert "graph shows symmetry" in out
    print("math directive v4 self-test passed")
