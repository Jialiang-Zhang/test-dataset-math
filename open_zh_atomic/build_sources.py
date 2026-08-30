#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

UA = "Jialiang-Zhang-open-zh-atomic-p3/1.0"
OLYMMATH_FILES = [
    ("https://raw.githubusercontent.com/RUCAIBox/OlymMATH/main/data/OlymMATH-EN-EASY.jsonl", "en", "easy"),
    ("https://raw.githubusercontent.com/RUCAIBox/OlymMATH/main/data/OlymMATH-EN-HARD.jsonl", "en", "hard"),
    ("https://raw.githubusercontent.com/RUCAIBox/OlymMATH/main/data/OlymMATH-ZH-EASY.jsonl", "zh", "easy"),
    ("https://raw.githubusercontent.com/RUCAIBox/OlymMATH/main/data/OlymMATH-ZH-HARD.jsonl", "zh", "hard"),
]
HARDMATH_URL = "https://raw.githubusercontent.com/sarahmart/HARDMath/main/data/HARDMath.json"
MATHODYSSEY_URL = "https://raw.githubusercontent.com/protagolabs/odyssey-math/main/final-odyssey-math-with-levels.jsonl"
MA_PROOFBENCH_URL = "https://raw.githubusercontent.com/OpenBMB/MA-ProofBench/main/benchmark/ma_proofbench.jsonl"
UG_BASE = "https://huggingface.co/datasets/UGMathBench/ugmathbench/resolve/main/data/{name}.json?download=true"
UG_SUBJECTS = [
    "Abstract_algebra", "Algebra", "Arithmetic", "Calculus_-_multivariable",
    "Calculus_-_single_variable", "Combinatorics", "Complex_analysis",
    "Differential_equations", "Financial_mathematics", "Geometry",
    "Linear_algebra", "Number_theory", "Probability", "Set_theory_and_logic",
    "Statistics", "Trigonometry",
]


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def fetch_text(url: str) -> str:
    return fetch_bytes(url).decode("utf-8")


def clean(value: Any) -> str:
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).replace("\x00", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def norm_hash(text: str) -> str:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text)).casefold())
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def answer_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return clean(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def base_row(*, source_family: str, source: str, problem: str, answer: str = "", solution: str = "",
             year: int | None = None, problem_no: str = "", subject: str = "", difficulty: str = "",
             language: str = "en", provenance_url: str = "", license_text: str = "", source_id: str = "") -> dict[str, Any]:
    return {
        "source_id": source_id,
        "problem": clean(problem),
        "answer": clean(answer),
        "solution": clean(solution),
        "source_family": source_family,
        "source": source,
        "year": year,
        "problem_no": problem_no,
        "subject": clean(subject),
        "difficulty": clean(difficulty),
        "language": language,
        "provenance_url": provenance_url,
        "provenance_tier": "P3",
        "license": license_text,
        "normalized_sha256": norm_hash(problem),
    }


def build_olymmath() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for url, language, difficulty in OLYMMATH_FILES:
        for line in fetch_text(url).splitlines():
            if not line.strip():
                continue
            src = json.loads(line)
            problem = clean(src.get("problem"))
            if not problem:
                continue
            sid = clean(src.get("unique_id")) or f"OlymMATH-{len(rows)}"
            rows.append(base_row(
                source_family="OlymMATH", source="OlymMATH", source_id=sid,
                problem=problem, answer=clean(src.get("answer")), solution=clean(src.get("solution")),
                subject=clean(src.get("subject")), difficulty=difficulty, language=language,
                provenance_url=url, license_text="MIT; preserve upstream copyright/license notice",
            ))
    return rows


def iter_records(obj: Any):
    if isinstance(obj, list):
        for item in obj:
            yield from iter_records(item)
    elif isinstance(obj, dict):
        keys = {str(k).casefold() for k in obj}
        if any(k in keys for k in ("question", "problem", "prompt")):
            yield obj
        else:
            for value in obj.values():
                yield from iter_records(value)


def boxed_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]{1,300})\}", solution)
    return matches[-1].strip() if matches else ""


def build_hardmath() -> list[dict[str, Any]]:
    raw = json.loads(fetch_text(HARDMATH_URL))
    rows: list[dict[str, Any]] = []
    for i, src in enumerate(iter_records(raw)):
        problem = clean(src.get("question") or src.get("problem") or src.get("prompt"))
        if not problem:
            continue
        solution = clean(src.get("solution") or src.get("worked_solution"))
        sid = clean(src.get("id") or src.get("problem_id")) or f"HARDMath-{i:04d}"
        rows.append(base_row(
            source_family="HARDMath", source="HARDMath", source_id=sid,
            problem=problem, answer=clean(src.get("answer")) or boxed_answer(solution), solution=solution,
            year=2024, subject=clean(src.get("problem_type") or src.get("type") or src.get("category") or "Applied Mathematics"),
            difficulty="graduate-level hard", language="en", provenance_url=HARDMATH_URL,
            license_text="MIT; preserve upstream copyright/license notice",
        ))
    return rows


def strip_odyssey(text: str) -> str:
    text = clean(text).replace("\\end{problem}", "").replace("\\noindent", "")
    text = re.sub(r"\\\\\s*$", "", text)
    return clean(text)


def build_mathodyssey() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(fetch_text(MATHODYSSEY_URL).splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if len(obj) != 1:
            raise RuntimeError(f"MathOdyssey line {line_no} shape changed")
        key, src = next(iter(obj.items()))
        problem = strip_odyssey(src.get("question", ""))
        if not problem:
            continue
        m = re.search(r"(\d+)$", str(key))
        pno = int(m.group(1)) if m else line_no
        rows.append(base_row(
            source_family="MathOdyssey", source=f"MathOdyssey / GAIC Math 2024·Problem {pno}", source_id=f"MathOdyssey-{pno}",
            problem=problem, answer=strip_odyssey(src.get("answer", "")), solution=strip_odyssey(src.get("reasoning", "")),
            year=2024, problem_no=str(pno), subject=clean(src.get("label")), difficulty=clean(src.get("level")),
            language="en", provenance_url=MATHODYSSEY_URL,
            license_text="CC BY-SA 4.0; preserve attribution and share-alike requirements",
        ))
    return rows


def build_ma_proofbench() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(fetch_text(MA_PROOFBENCH_URL).splitlines(), 1):
        if not line.strip():
            continue
        src = json.loads(line)
        problem = clean(src.get("informal_statement"))
        if not problem:
            continue
        rid = src.get("id", line_no)
        row = base_row(
            source_family="MA-ProofBench", source=f"MA-ProofBench·{clean(src.get('split'))}·{rid}", source_id=f"MA-ProofBench-{rid}",
            problem=problem, year=2026, problem_no=str(rid), subject=clean(src.get("topic")), difficulty=clean(src.get("split")),
            language="en", provenance_url=MA_PROOFBENCH_URL, license_text="MIT",
        )
        row.update({"formal_statement": clean(src.get("formal_statement")), "tag": clean(src.get("tag")), "lean_version": clean(src.get("version"))})
        rows.append(row)
    return rows


def load_json_array(url: str) -> list[dict[str, Any]]:
    raw = json.loads(fetch_text(url))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "test", "examples"):
            if isinstance(raw.get(key), list):
                return raw[key]
    raise RuntimeError(f"UGMathBench source shape changed: {url}")


def build_ugmathbench() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_count = 0
    for file_subject in UG_SUBJECTS:
        url = UG_BASE.format(name=urllib.parse.quote(file_subject, safe="_-"))
        for src in load_json_array(url):
            base_count += 1
            subject = clean(src.get("subject")) or file_subject
            base_id = clean(src.get("id")) or f"{file_subject}-{base_count:05d}"
            for version in (1, 2, 3):
                problem = clean(src.get(f"problem_v{version}"))
                if not problem:
                    raise RuntimeError(f"UGMathBench missing problem_v{version}: {base_id}")
                row = base_row(
                    source_family="UGMathBench", source=f"UGMathBench·{subject}·{base_id}·v{version}",
                    source_id=f"UGMathBench-{base_id}-v{version}", problem=problem,
                    answer=answer_string(src.get(f"answer_v{version}")), year=2025,
                    problem_no=f"{base_id}-v{version}", subject=subject, difficulty=clean(src.get("level")), language="en",
                    provenance_url=url, license_text="GPL-3.0",
                )
                row.update({
                    "variant_of": base_id, "variant_index": version, "topic": clean(src.get("topic")),
                    "subtopic": clean(src.get("subtopic")), "keywords": src.get("keywords") or [],
                    "answer_type": src.get(f"answer_type_v{version}") or [], "options": src.get(f"options_v{version}") or [],
                })
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="open_zh_atomic/work/raw")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = {
        "OlymMATH": build_olymmath(),
        "HARDMath": build_hardmath(),
        "MathOdyssey": build_mathodyssey(),
        "MA-ProofBench": build_ma_proofbench(),
        "UGMathBench": build_ugmathbench(),
    }
    expected = {"OlymMATH": (390, 410), "HARDMath": (1000, 1100), "MathOdyssey": (380, 400), "MA-ProofBench": (200, 200), "UGMathBench": (15000, 15300)}
    for name, rows in groups.items():
        lo, hi = expected[name]
        if not lo <= len(rows) <= hi:
            raise RuntimeError(f"{name} unexpected rows: {len(rows)}")
        write_jsonl(out / f"{name.lower().replace('-', '_')}.jsonl", rows)

    all_rows = [row for rows in groups.values() for row in rows]
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in all_rows:
        h = row["normalized_sha256"]
        if h in seen:
            continue
        seen.add(h)
        row = dict(row)
        row["parent_id"] = f"P3-{len(unique):05d}-{h[:12]}"
        unique.append(row)

    write_jsonl(out / "all_open_p3_raw.jsonl", all_rows)
    write_jsonl(out / "all_open_p3_exact_unique.jsonl", unique)
    report = {
        "raw_rows": len(all_rows), "exact_unique_rows": len(unique),
        "duplicates_removed": len(all_rows) - len(unique),
        "by_source_raw": dict(Counter(r["source_family"] for r in all_rows)),
        "by_source_unique": dict(Counter(r["source_family"] for r in unique)),
        "licenses": {"OlymMATH": "MIT", "HARDMath": "MIT", "MathOdyssey": "CC BY-SA 4.0", "MA-ProofBench": "MIT", "UGMathBench": "GPL-3.0"},
    }
    (out / "source_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
