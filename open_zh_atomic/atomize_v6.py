#!/usr/bin/env python3
from __future__ import annotations

import re
import atomize_v4
import atomize_v5
import atomize as core

STRUCTURED_OBJECT_RE = re.compile(
    r"(?i)(?:ordered\s+pair|one\s+pair|coordinates?\s+of\s+the\s+point|rectangular\s+coordinates|"
    r"find\s+(?:the\s+)?matrix|find\s+(?:the\s+)?vector|components?\s+of|"
    r"in\s+the\s+form\s+\$?u\+wt|piecewise[- ]defined\s+(?:linear\s+)?function|"
    r"rewrite\s+the\s+following\s+using\s+a\s+single\s+exponent)|"
    r"(?:有序对|点的坐标|直角坐标|求矩阵|求向量|分量|写成.*u\+wt|分段函数)"
)
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
    if len(matches) < 2 or len(matches) != text.count('[ANS]'):
        return None
    common = text[:matches[0].start()].rstrip(' ,，;；')
    common = re.sub(r"(?i)\b(?:and|then)\s*$", "", common).rstrip(' ,，;；')
    out = []
    for i, m in enumerate(matches, 1):
        clause = f"{m.group('name')} {m.group('link')} [ANS]"
        tail = text[m.end():]
        punct = re.match(r"\s*([.,;])", tail)
        if punct:
            clause += punct.group(1)
        child = f"{common}, {clause}".strip(' ,') if common else clause
        out.append((f"field{i}", child))
    return out if len(out) >= 2 else None


def split_atomic_problem_v6(text: str):
    # First split container/list structure so named fields cannot capture fields from
    # multiple objects at once. These functions are the conservative structural rules
    # already validated in v4/v5.
    structural = (
        ('explicit_parts', core.split_explicit_parts),
        ('conjoined_tasks', core.split_conjoined_tasks),
        ('separate_task_sentences', core.split_task_sentences),
        ('multiple_question_sentences', core.split_question_marks),
        ('answer_array_rows', atomize_v4.split_answer_array_rows),
        ('answer_lines', atomize_v4.split_answer_lines),
        ('numbered_answer_items', atomize_v5.split_numbered_answer_items),
        ('inline_lettered_items', atomize_v5.split_inline_lettered_items),
        ('labelled_definition_items', atomize_v5.split_labelled_definition_items),
        ('repeated_for_items', atomize_v5.split_repeated_for_items),
    )
    for reason, fn in structural:
        parts = fn(text)
        if parts:
            return parts, reason

    # Once one object/item remains, split genuinely distinct named targets while
    # preserving the complete object/equation context in each child.
    named = split_named_fields(text)
    if named:
        return named, 'named_answer_fields'

    # Generic answer-sentence splitting is deliberately last because it is less
    # semantically specific than the structural and named-field rules above.
    extra = atomize_v5.split_answer_sentences(text)
    if extra:
        return extra, 'answer_sentences'
    return [('q1', text.strip())], 'unsplit'


core.split_atomic_problem = split_atomic_problem_v6


def self_test() -> None:
    whole = (
        r"Find the slope and $y$ intercept for each of the following lines.\n"
        r"For $3y-12x=6$, slope=[ANS] and y intercept=[ANS]. "
        r"For $y=8x+3$, slope=[ANS] and y intercept=[ANS]."
    )
    first, first_reason = core.split_atomic_problem(whole)
    assert len(first) == 2 and first_reason == 'repeated_for_items', (first_reason, first)
    children = []
    for _, item in first:
        sub, reason = core.split_atomic_problem(item)
        assert len(sub) == 2 and reason == 'named_answer_fields', (reason, sub)
        children.extend(sub)
    assert sum('$3y-12x=6$' in x[1] for x in children) == 2, children
    assert sum('$y=8x+3$' in x[1] for x in children) == 2, children

    pair = r"Find one pair $(x,y)$. $x=$ [ANS] $y=$ [ANS]"
    parts2, reason2 = core.split_atomic_problem(pair)
    assert len(parts2) == 1, (reason2, parts2)

    piece = r"Write a piecewise-defined linear function $C(m)$. [ANS] if [ANS] <= m <= [ANS], [ANS] if m > [ANS]"
    parts3, reason3 = core.split_atomic_problem(piece)
    assert len(parts3) == 1, (reason3, parts3)


if __name__ == '__main__':
    self_test()
    raise SystemExit(core.main())
