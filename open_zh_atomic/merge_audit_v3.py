#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import atomize_v3  # installs guarded question detection + full math-environment masking
import atomize as atom
from glossary import audit as audit_glossary
from tex_protection_v3 import (
    extract_primary_math, critical_tex_sequence, has_english_prose, mask_primary_math,
)

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")


def norm_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", "", text.casefold()).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="open_zh_atomic/work/prepared.jsonl")
    ap.add_argument("--shards-dir", default="open_zh_atomic/work/shards")
    ap.add_argument("--output-dir", default="open_zh_atomic/output")
    args = ap.parse_args()

    prepared = load_jsonl(Path(args.prepared))
    expected = {int(r["global_atomic_index"]): r for r in prepared}
    if len(expected) != len(prepared):
        raise RuntimeError("duplicate prepared global index")

    shard_files = sorted(Path(args.shards_dir).rglob("shard-*.jsonl"))
    if not shard_files:
        raise RuntimeError("no translated shard files")
    got: dict[int, dict[str, Any]] = {}
    for path in shard_files:
        for row in load_jsonl(path):
            idx = int(row["global_atomic_index"])
            if idx in got:
                raise RuntimeError(f"duplicate translated index {idx}")
            got[idx] = row
    if set(got) != set(expected):
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise RuntimeError(f"coverage mismatch missing={missing[:20]} extra={extra[:20]}")

    rows = [got[i] for i in sorted(got)]
    issues: list[dict[str, Any]] = []
    parents: set[str] = set()
    by_source = Counter()
    methods = Counter()
    split_children = 0
    formula_only_rows = 0

    for row in rows:
        idx = int(row["global_atomic_index"])
        exp = expected[idx]
        source = str(row.get("problem_atomic_original") or "")
        zh = str(row.get("problem_zh") or "").strip()
        row_issues: list[str] = []

        if str(row.get("id")) != str(exp.get("id")):
            row_issues.append("id_mismatch")
        if str(row.get("parent_id")) != str(exp.get("parent_id")):
            row_issues.append("parent_id_mismatch")
        if source != str(exp.get("atomic_source_problem") or ""):
            row_issues.append("atomic_source_mismatch")
        if not zh:
            row_issues.append("empty_problem")
        else:
            parts, reason = atom.split_atomic_problem(zh)
            if len(parts) > 1:
                row_issues.append(f"residual_multiquestion:{reason}:{len(parts)}")
            if extract_primary_math(source) != extract_primary_math(zh):
                row_issues.append("primary_math_or_environment_not_preserved")
            if critical_tex_sequence(source) != critical_tex_sequence(zh):
                row_issues.append("bare_tex_math_skeleton_not_preserved")
            missing_terms = audit_glossary(mask_primary_math(source), zh)
            if missing_terms:
                row_issues.append("math_glossary_missing:" + ",".join(missing_terms))
            needs_translation = has_english_prose(source)
            if needs_translation:
                if len(CJK_RE.findall(zh)) < 2:
                    row_issues.append("insufficient_chinese")
            else:
                formula_only_rows += 1

        is_split = int(exp.get("atomic_count_from_parent") or 1) > 1
        if is_split:
            split_children += 1
            if exp.get("answer_scope") != "parent_aggregate_not_attached":
                row_issues.append("bad_answer_scope")
            if exp.get("answer") or exp.get("solution") or row.get("answer") or row.get("solution"):
                row_issues.append("split_child_inherited_parent_answer")

        if not row.get("translation_primary_math_preserved"):
            row_issues.append("worker_primary_math_flag_false")
        if not row.get("translation_bare_tex_skeleton_preserved"):
            row_issues.append("worker_bare_tex_flag_false")
        if not row.get("translation_math_glossary_preserved"):
            row_issues.append("worker_glossary_flag_false")

        if row_issues:
            issues.append({
                "idx": idx, "id": row.get("id"), "parent_id": row.get("parent_id"),
                "issues": row_issues, "problem": zh, "problem_original": source,
            })
        parents.add(str(row.get("parent_id")))
        by_source[str(row.get("source_family"))] += 1
        methods[str(row.get("translation_method"))] += 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "needs_review.jsonl", issues)
    residual = sum(any(x.startswith("residual_multiquestion") for x in i["issues"]) for i in issues)
    report = {
        "prepared_atomic_rows": len(prepared),
        "translated_atomic_rows": len(rows),
        "unique_parent_rows_covered": len(parents),
        "issue_rows": len(issues),
        "residual_multiquestion_rows": residual,
        "split_children_answer_detachment_audited": split_children,
        "formula_only_rows": formula_only_rows,
        "by_source_family": dict(by_source),
        "translation_methods": dict(methods),
        "unique_chinese_problem_hashes": len({norm_hash(str(r.get("problem_zh") or "")) for r in rows}),
        "audit_contract": {
            "one_question_per_row": True,
            "guarded_interrogative_detection": True,
            "full_latex_math_environments_preserved": True,
            "delimited_math_preserved": True,
            "bare_tex_math_skeleton_preserved": True,
            "formula_only_rows_may_remain_formula_only": True,
            "english_natural_language_translated_to_chinese": True,
            "high_confidence_math_glossary_preserved": True,
            "split_children_do_not_inherit_parent_answers": True,
        },
    }
    (out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if issues:
        print(json.dumps(report, ensure_ascii=False))
        print(json.dumps(issues[:20], ensure_ascii=False, indent=2))
        raise RuntimeError(f"v3 audit failed for {len(issues)} rows")

    final: list[dict[str, Any]] = []
    for out_idx, row in enumerate(rows):
        final.append({
            "idx": out_idx,
            "problem": row["problem_zh"],
            "problem_original": row["problem_atomic_original"],
            "subject": row.get("subject", ""),
            "source": row.get("source", ""),
            "source_family": row.get("source_family", ""),
            "year": row.get("year"),
            "difficulty": row.get("difficulty", ""),
            "parent_id": row.get("parent_id", ""),
            "normalized_parent_sha256": row.get("normalized_sha256", ""),
            "atomic_index": row.get("atomic_index", 1),
            "atomic_count_from_parent": row.get("atomic_count_from_parent", 1),
            "atomic_label_path": row.get("atomic_label_path", []),
            "split_reason_chain": row.get("split_reason_chain", []),
            "source_language": row.get("source_language_before_translation", ""),
            "provenance_url": row.get("provenance_url", ""),
            "provenance_tier": "P3",
            "license": row.get("license", ""),
            "translation_method": row.get("translation_method", ""),
            "atomicity_status": "passed",
            "translation_status": "passed",
        })
    write_jsonl(out / "questions_zh_atomic_p3.jsonl", final)
    write_jsonl(out / "questions_zh_atomic_p3_min.jsonl", [
        {"idx": r["idx"], "problem": r["problem"], "subject": r["subject"],
         "source_family": r["source_family"], "parent_id": r["parent_id"],
         "normalized_parent_sha256": r["normalized_parent_sha256"]}
        for r in final
    ])
    (out / "README.md").write_text(
        f"# Open P3 中文原子数学题库 v3\n\n"
        f"- 唯一父题：{len(parents)}\n- 中文原子题：{len(final)}\n- 纯公式题：{formula_only_rows}\n- 审计问题：0\n"
        "- 每条 JSONL 只有一个问题；多问递归拆分。\n"
        "- equation/align/array/cases/matrix 等 LaTeX 环境与普通数学定界公式均不可翻译。\n"
        "- 裸 TeX 数学骨架独立二次审计；只有自然语言片段进入 Marian 翻译。\n"
        "- 高置信数学术语使用受控中文词汇表。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
