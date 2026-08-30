#!/usr/bin/env python3
from __future__ import annotations

import re
import atomize_v5  # installs v5 rules
import atomize as core

BASE_SPLIT = core.split_atomic_problem
STRUCTURED_OBJECT_RE = re.compile(
    r"(?i)(?:ordered\s+pair|one\s+pair|coordinates?\s+of\s+the\s+point|rectangular\s+coordinates|"
    r"find\s+(?:the\s+)?matrix|find\s+(?:the\s+)?vector|components?\s+of|"
    r"in\s+the\s+form\s+\$?u\+wt|piecewise[- ]defined\s+(?:linear\s+)?function|"
    r"rewrite\s+the\s+following\s+using\s+a\s+single\s+exponent)|"
    r"(?:有序对|点的坐标|直角坐标|求矩阵|求向量|分量|写成.*u\+wt|分段函数)"
)
# A field name immediately followed by = [ANS] / is [ANS]. Keep the name short so
# prose far away from the blank cannot be swallowed into a field label.
FIELD_RE = re.compile(
    r"(?i)(?P<name>(?:the\s+)?(?:slope|y\s*intercept|x\s*intercept|coefficient|exponent|"
    r"mean|median|mode|variance|standard\s+deviation|probability|real\s+part|imaginary\s+part|"
    r"magnitude|argument|radius|diameter|area|volume|length|width|height|domain|range))"
    r"\s*(?P<link>=|\bis\b)\s*\[ANS\]"
)


def split_named_fields(text: str):
    if text.count('[ANS]') < 2 or STRUCTURED_OBJECT_RE.search(core.mask_math(text)):
        return None
    matches = list(FIELD_RE.finditer(text))
    if len(matches) < 2:
        return None
    # Every answer slot in this leaf must belong to one recognized field; otherwise a
    # separate unnamed component may be part of a single structured response.
    if len(matches) != text.count('[ANS]'):
        return None
    common = text[:matches[0].start()].rstrip(' ,，;；')
    # Remove a trailing conjunction from common prefix if present.
    common = re.sub(r"(?i)\b(?:and|then)\s*$", "", common).rstrip(' ,，;；')
    out = []
    for i, m in enumerate(matches, 1):
        clause = f"{m.group('name')} {m.group('link')} [ANS]"
        # Preserve terminal punctuation from the local source when easy to identify.
        tail = text[m.end():]
        punct = re.match(r"\s*([.,;])", tail)
        if punct:
            clause += punct.group(1)
        child = f"{common}, {clause}".strip(' ,') if common else clause
        out.append((f"field{i}", child))
    return out if len(out) >= 2 else None


def split_atomic_problem_v6(text: str):
    # Named fields must run before v5's generic answer_sentences, otherwise the second
    # field can lose the shared object/equation context.
    named = split_named_fields(text)
    if named:
        return named, 'named_answer_fields'
    return BASE_SPLIT(text)

core.split_atomic_problem = split_atomic_problem_v6


def self_test() -> None:
    slope = r"Find the slope and $y$ intercept for each of the following lines.\nFor $3y-12x=6$, slope=[ANS] and y intercept=[ANS]."
    parts, reason = core.split_atomic_problem(slope)
    assert len(parts) == 2 and reason == 'named_answer_fields', (reason, parts)
    assert all('$3y-12x=6$' in p[1] for p in parts), parts
    assert 'slope' in parts[0][1].lower() and 'intercept' not in parts[0][1].lower().split('[ANS]')[0].split(',')[-1], parts
    assert 'y intercept' in parts[1][1].lower(), parts

    pair = r"Find one pair $(x,y)$. $x=$ [ANS] $y=$ [ANS]"
    parts2, reason2 = core.split_atomic_problem(pair)
    assert len(parts2) == 1, (reason2, parts2)

    piece = r"Write a piecewise-defined linear function $C(m)$. [ANS] if [ANS] <= m <= [ANS], [ANS] if m > [ANS]"
    parts3, reason3 = core.split_atomic_problem(piece)
    assert len(parts3) == 1, (reason3, parts3)


if __name__ == '__main__':
    self_test()
    raise SystemExit(core.main())
