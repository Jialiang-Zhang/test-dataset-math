#!/usr/bin/env python3
from __future__ import annotations

import re
import atomize_v3  # guarded interrogatives + TeX-aware math masking
import atomize as core
from tex_protection_v3 import primary_math_spans

BASE_SPLIT = core.split_atomic_problem
PAIR_GUARD_RE = re.compile(
    r"(?i)\b(?:pair|ordered\s+pair|coordinates?|components?|real\s+and\s+imaginary\s+parts?)\b|"
    r"(?:有序对|坐标|分量|实部和虚部|实部与虚部)"
)
GLOBAL_CONTEXT_RE = re.compile(
    r"(?i)(?:following|for\s+each|in\s+this\s+problem|choose\s+true\s+or\s+false|"
    r"^\s*(?:let|suppose|assume|given|define|consider)\b)|"
    r"(?:下列|以下|每个|各项|各题|判断下列|设|假设|已知|定义|考虑)"
)
TABLE_LIST_RE = re.compile(
    r"(?i)(?:each\s+of\s+the\s+following|following\s+(?:elements|quantities|questions|statements|values|expressions)|"
    r"choose\s+.*following|for\s+each)|(?:以下各|下列各|下列问题|下列命题|逐项|每一项)"
)
ARRAY_RE = re.compile(r"\\begin\{array\}\{[^{}]*\}(.*?)\\end\{array\}", re.S)
TASK_LINE_RE = re.compile(
    r"(?i)^\s*(?:find|determine|compute|calculate|evaluate|solve|prove|show|write|express|choose|select|"
    r"classify|construct|describe|verify|derive|give|decide|state|explain|identify)\b|"
    r"^\s*(?:求|确定|计算|求值|求解|证明|写出|表示|选择|判断|分类|构造|描述|验证|推导|给出|说明|识别)"
)


def _ans_positions_inside_one_math_environment(text: str) -> bool:
    positions = [m.start() for m in re.finditer(r"\[ANS\]", text)]
    if len(positions) < 2:
        return False
    spans = primary_math_spans(text)
    containing = []
    for pos in positions:
        hit = next((i for i, span in enumerate(spans) if span.start <= pos < span.end), None)
        containing.append(hit)
    return containing[0] is not None and all(x == containing[0] for x in containing)


def _common_suffix(lines: list[str], last_answer_line: int) -> list[str]:
    suffix = lines[last_answer_line + 1:]
    if not suffix:
        return []
    text = "\n".join(suffix).strip()
    if not text or "[ANS]" in text:
        return []
    # Do not append a new independent task to every child.
    if core.has_task(text):
        return []
    return suffix


def split_answer_lines(text: str):
    if text.count("[ANS]") < 2:
        return None
    if _ans_positions_inside_one_math_environment(text):
        return None
    lines = text.splitlines()
    answer_idx = [i for i, line in enumerate(lines) if "[ANS]" in line]
    if len(answer_idx) < 2:
        return None

    # A single request whose answer is explicitly one pair/vector/coordinate tuple is
    # one question even when the UI gives one blank per component.
    if PAIR_GUARD_RE.search(core.mask_math(text)):
        independent_task_lines = sum(bool(TASK_LINE_RE.search(core.mask_math(lines[i]))) for i in answer_idx)
        if independent_task_lines < 2:
            return None

    first_ans = answer_idx[0]
    prefix_lines = lines[:first_ans]
    prefix_text = "\n".join(prefix_lines).strip()
    shared_lines: list[str] = []
    first_block_start = 0

    if prefix_lines:
        task_lines = [i for i, line in enumerate(prefix_lines) if TASK_LINE_RE.search(core.mask_math(line))]
        if GLOBAL_CONTEXT_RE.search(core.mask_math(prefix_text)):
            shared_lines = prefix_lines
            first_block_start = first_ans
        elif task_lines:
            first_block_start = task_lines[-1]
            shared_lines = prefix_lines[:first_block_start]
        else:
            # Pure setup such as "Let f(r)=..." is shared by all requested values.
            shared_lines = prefix_lines
            first_block_start = first_ans

    suffix = _common_suffix(lines, answer_idx[-1])
    children = []
    previous_answer = None
    for j, idx in enumerate(answer_idx, start=1):
        if j == 1:
            start = first_block_start
        else:
            start = int(previous_answer) + 1
        block = lines[start:idx + 1]
        if not "\n".join(block).strip():
            previous_answer = idx
            continue
        child_lines = []
        if shared_lines:
            child_lines.extend(shared_lines)
        child_lines.extend(block)
        if suffix:
            child_lines.extend(suffix)
        child = "\n".join(child_lines).strip()
        if child and child != text.strip():
            children.append((f"ansline{j}", child))
        previous_answer = idx
    return children if len(children) >= 2 else None


def _strip_dollar_around_environment(text: str, start: int, end: int):
    left, right = start, end
    i = left - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    j = right
    while j < len(text) and text[j].isspace():
        j += 1
    if i >= 0 and j < len(text) and text[i] == "$" and text[j] == "$":
        return i, j + 1
    return start, end


def split_answer_array_rows(text: str):
    if text.count("[ANS]") < 2:
        return None
    for match in ARRAY_RE.finditer(text):
        body = match.group(1)
        if body.count("[ANS]") < 2:
            continue
        prefix_probe = text[max(0, match.start() - 1800):match.start()]
        if not TABLE_LIST_RE.search(core.mask_math(prefix_probe)):
            continue
        rows = re.split(r"\\\\", body)
        answer_rows = []
        for row in rows:
            if "[ANS]" not in row:
                continue
            clean = re.sub(r"\\hline\b", " ", row)
            clean = re.sub(r"\s+", " ", clean).strip(" &\t\r\n")
            if clean:
                answer_rows.append(clean)
        if len(answer_rows) < 2:
            continue

        full_start, full_end = _strip_dollar_around_environment(text, match.start(), match.end())
        prefix = text[:full_start].strip()
        suffix = text[full_end:].strip()
        if suffix and core.has_task(suffix):
            suffix = ""
        children = []
        for i, row in enumerate(answer_rows, start=1):
            # Ampersands in an array separate cells. Replacing them by spacing keeps a
            # multi-blank row as one answer unit instead of inventing extra questions.
            rendered = re.sub(r"\s*&\s*", r" \\quad ", row)
            child = f"{prefix}\n${rendered}$".strip()
            if suffix:
                child = f"{child}\n{suffix}".strip()
            children.append((f"ansrow{i}", child))
        return children
    return None


def split_atomic_problem_v4(text: str):
    parts, reason = BASE_SPLIT(text)
    if len(parts) > 1:
        return parts, reason
    for reason2, fn in (
        ("answer_array_rows", split_answer_array_rows),
        ("answer_lines", split_answer_lines),
    ):
        extra = fn(text)
        if extra:
            return extra, reason2
    return parts, reason


core.split_atomic_problem = split_atomic_problem_v4


def self_test() -> None:
    # Four derivative outputs are four asks.
    d = r"""Let $f(r)=3 \sqrt{r}+6 \sqrt[3]{r}$.
$f'(r)=$ [ANS]
$f'(3)=$ [ANS],
$f''(r)=$ [ANS]
$f''(3)=$ [ANS]"""
    parts, reason = core.split_atomic_problem(d)
    assert len(parts) == 4 and reason == "answer_lines", (reason, parts)

    # One ordered pair is one question even if the UI has two component blanks.
    pair = r"""Find one pair $(x,y)$ satisfying the conditions.
$x=$ [ANS]
$y=$ [ANS]"""
    parts2, reason2 = core.split_atomic_problem(pair)
    assert len(parts2) == 1, (reason2, parts2)

    # One representation u+wt remains one question despite two blanks on the same row.
    one = r"Express $t^5$ in the form $u+wt$.\n$t^5$: [ANS] $+$ [ANS] $t$"
    parts3, reason3 = core.split_atomic_problem(one)
    assert len(parts3) == 1, (reason3, parts3)

    # A list table under "following" splits by table row, not by individual blank.
    arr = r"""Express each of the following elements.
$\begin{array}{cc}\hline (x7)^2 & [ANS] \\ \hline (x8)^3 & [ANS] \\ \hline (x7)^{-1} & [ANS] \\ \hline \end{array}$"""
    parts4, reason4 = core.split_atomic_problem(arr)
    assert len(parts4) == 3 and reason4 == "answer_array_rows", (reason4, parts4)


if __name__ == "__main__":
    self_test()
    raise SystemExit(core.main())
