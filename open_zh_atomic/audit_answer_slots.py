#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PAIR_RE = re.compile(
    r"(?i)\b(?:pair|ordered\s+pair|coordinates?|components?|real\s+and\s+imaginary\s+parts?|"
    r"vector|point\s+\([^)]*,[^)]*\))\b|(?:有序对|坐标|分量|实部和虚部|实部与虚部|向量)"
)
LIST_HINT_RE = re.compile(
    r"(?i)(?:each\s+of\s+the\s+following|following\s+(?:elements|questions|statements|values|expressions|quantities)|"
    r"for\s+each|choose\s+.*following|true\s+or\s+false)|(?:以下各|下列各|下列问题|下列命题|逐项|每一项|判断下列)"
)
TASK_RE = re.compile(
    r"(?i)\b(?:find|determine|compute|calculate|evaluate|solve|prove|show|write|express|choose|select|"
    r"classify|construct|describe|verify|derive|give|decide|state|explain|identify)\b|"
    r"(?:求|确定|计算|求值|求解|证明|写出|表示|选择|判断|分类|构造|描述|验证|推导|给出|说明|识别)"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")


def answer_type_count(row: dict[str, Any]) -> int | None:
    value = row.get("answer_type")
    if isinstance(value, list):
        return len(value)
    if value in (None, "", []):
        return None
    return 1


def category(row: dict[str, Any]) -> tuple[str, list[str]]:
    text = str(row.get("atomic_source_problem") or "")
    ans = text.count("[ANS]")
    lines = text.splitlines()
    ans_lines = [line for line in lines if "[ANS]" in line]
    atc = answer_type_count(row)
    flags: list[str] = []
    if ans <= 1:
        return "single_slot", flags
    if len(ans_lines) >= 2:
        flags.append("multiple_answer_lines")
    if len(ans_lines) == 1:
        flags.append("same_answer_line")
    if PAIR_RE.search(text):
        flags.append("pair_component_hint")
    if LIST_HINT_RE.search(text):
        flags.append("list_hint")
    task_count = len(TASK_RE.findall(text))
    if task_count >= 2:
        flags.append(f"multiple_task_words:{task_count}")
    if atc is not None:
        flags.append(f"answer_type_count:{atc}")
        if atc == 1:
            flags.append("single_structured_answer_metadata")
        elif atc > 1:
            flags.append("multi_output_metadata")

    # High-confidence acceptable: source metadata says exactly one structured answer,
    # or a pair/vector/component request with no independent list/task evidence.
    if atc == 1:
        return "likely_single_structured_answer", flags
    if PAIR_RE.search(text) and not LIST_HINT_RE.search(text) and task_count <= 1:
        return "likely_single_component_answer", flags

    # High-confidence suspicious: multiple output metadata survived as one leaf, or a
    # list prompt/table still contains more than one answer slot.
    if atc is not None and atc > 1:
        return "suspicious_multi_output_metadata", flags
    if LIST_HINT_RE.search(text):
        return "suspicious_list_with_multi_slots", flags
    if len(ans_lines) >= 2:
        return "suspicious_multiple_answer_lines", flags

    # Same-line multi blanks without decisive metadata can be a polynomial/vector/etc.
    return "ambiguous_same_line_multi_slot", flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--suspects", required=True)
    ap.add_argument("--ambiguous", required=True)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    multi = [r for r in rows if str(r.get("atomic_source_problem") or "").count("[ANS]") >= 2]
    counts = Counter()
    flags = Counter()
    by_source = Counter()
    suspects: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for row in multi:
        cat, row_flags = category(row)
        counts[cat] += 1
        by_source[str(row.get("source_family") or "unknown")] += 1
        flags.update(row_flags)
        record = {
            "id": row.get("id"),
            "source_id": row.get("source_id"),
            "source_family": row.get("source_family"),
            "subject": row.get("subject"),
            "answer_type": row.get("answer_type"),
            "ans_count": str(row.get("atomic_source_problem") or "").count("[ANS]"),
            "answer_line_count": sum("[ANS]" in x for x in str(row.get("atomic_source_problem") or "").splitlines()),
            "category": cat,
            "flags": row_flags,
            "problem": row.get("atomic_source_problem"),
        }
        if cat.startswith("suspicious_"):
            suspects.append(record)
        elif cat.startswith("ambiguous_"):
            ambiguous.append(record)

    report = {
        "total_atomic_rows": len(rows),
        "multi_ans_atomic_rows": len(multi),
        "category_counts": dict(counts),
        "flag_counts": dict(flags),
        "multi_ans_by_source": dict(by_source),
        "high_confidence_suspect_rows": len(suspects),
        "ambiguous_rows": len(ambiguous),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(Path(args.suspects), suspects)
    write_jsonl(Path(args.ambiguous), ambiguous)
    print(json.dumps(report, ensure_ascii=False))
    print(json.dumps(suspects[:12], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
