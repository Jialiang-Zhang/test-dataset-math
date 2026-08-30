#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import MarianMTModel, MarianTokenizer

from glossary import PATTERN as GLOSSARY_RE, rule_for, audit as audit_glossary
from tex_protection_v3 import (
    PLACEHOLDER_RE, NUMBER_RE, TEX_COMMAND_RE, SINGLE_VAR_RE, ALLCAPS_RE,
    primary_math_spans, extract_primary_math, critical_tex_sequence,
    natural_words_outside_math, has_english_prose,
)

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
RAW_TOKEN_RE = re.compile(
    r"\[ANS\]|(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?|"
    r"\\(?:[A-Za-z]+|.)|(?<![A-Za-z])[A-Z]{2,8}(?![A-Za-z])|(?<![A-Za-z])[A-Za-z](?![A-Za-z])"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text))


def append_gap(text: str, parts: list[tuple[str, str | int]], units: list[str]) -> None:
    if not text:
        return
    if natural_words_outside_math(text):
        idx = len(units)
        units.append(text)
        parts.append(("trans", idx))
    else:
        parts.append(("raw", text))


def append_nonmath(text: str, parts: list[tuple[str, str | int]], units: list[str]) -> None:
    """Translate only natural-language islands; preserve TeX, variables and literals."""
    candidates: list[tuple[int, int, int, str, str]] = []
    # Priority 0 glossary terms: replace with canonical Chinese and never send to MT.
    for m in GLOSSARY_RE.finditer(text):
        candidates.append((m.start(), m.end(), 0, "glossary", rule_for(m).zh))
    # Priority 1 raw math-ish atoms outside normal delimiters/environments.
    for m in RAW_TOKEN_RE.finditer(text):
        candidates.append((m.start(), m.end(), 1, "raw", m.group(0)))
    candidates.sort(key=lambda x: (x[0], x[2], -(x[1] - x[0])))

    cursor = 0
    for start, end, _, kind, value in candidates:
        if start < cursor:
            continue
        if start > cursor:
            append_gap(text[cursor:start], parts, units)
        parts.append(("raw", value))
        cursor = end
    if cursor < len(text):
        append_gap(text[cursor:], parts, units)


def build_structures(texts: list[str]) -> tuple[list[list[tuple[str, str | int]]], list[str]]:
    structures: list[list[tuple[str, str | int]]] = []
    units: list[str] = []
    for text in texts:
        parts: list[tuple[str, str | int]] = []
        cursor = 0
        for span in primary_math_spans(text):
            if span.start > cursor:
                append_nonmath(text[cursor:span.start], parts, units)
            parts.append(("raw", text[span.start:span.end]))
            cursor = span.end
        if cursor < len(text):
            append_nonmath(text[cursor:], parts, units)
        structures.append(parts)
    return structures, units


def sanitize(source: str, translated: str) -> str:
    # Translation units are prose-only by construction. Any new TeX syntax is spurious.
    out = translated
    out = out.replace("$", "").replace("[ANS]", "")
    out = out.replace(r"\[", "").replace(r"\]", "").replace(r"\(", "").replace(r"\)", "")
    out = re.sub(r"\\(?:[A-Za-z]+|.)", "", out)
    if "?" not in source and "？" not in source:
        out = out.replace("?", "").replace("？", "")
    if "!" not in source and "！" not in source:
        out = out.replace("!", "").replace("！", "")
    return out.strip()


def translate_cached(
    units: list[str], tokenizer: MarianTokenizer, model: MarianMTModel,
    batch_size: int, cache: dict[str, str]
) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for unit in units:
        if unit not in cache and unit not in seen:
            seen.add(unit)
            missing.append(unit)
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.inference_mode():
            generated = model.generate(**encoded, num_beams=1, do_sample=False, max_new_tokens=512)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for src, dst in zip(batch, decoded):
            cache[src] = sanitize(src, dst)
    return [cache[u] for u in units]


def reassemble(structures: list[list[tuple[str, str | int]]], translated: list[str]) -> list[str]:
    out: list[str] = []
    for parts in structures:
        out.append("".join(str(value) if kind == "raw" else translated[int(value)] for kind, value in parts).strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="open_zh_atomic/work/prepared.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--shard-count", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--model", default="Helsinki-NLP/opus-mt-en-zh")
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
    all_rows = load_jsonl(Path(args.input))
    selected = [r for r in all_rows if int(r["global_atomic_index"]) % args.shard_count == args.shard_id]
    if not selected:
        raise RuntimeError(f"empty shard {args.shard_id}/{args.shard_count}")

    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model)
    model.eval()
    cache: dict[str, str] = {}
    completed: list[dict[str, Any]] = []
    problem_batch = 32

    for start in range(0, len(selected), problem_batch):
        batch_rows = selected[start:start + problem_batch]
        source_texts = [str(r["atomic_source_problem"]) for r in batch_rows]
        structures, units = build_structures(source_texts)
        translated_units = translate_cached(units, tokenizer, model, args.batch_size, cache)
        zh_texts = reassemble(structures, translated_units)

        for row, source, zh in zip(batch_rows, source_texts, zh_texts):
            if not zh:
                raise RuntimeError(f"empty translation {row['id']}")
            if extract_primary_math(source) != extract_primary_math(zh):
                raise RuntimeError(f"primary math/environment preservation {row['id']}")
            if critical_tex_sequence(source) != critical_tex_sequence(zh):
                print(json.dumps({
                    "kind": "critical_tex_sequence_failure", "id": row.get("id"),
                    "source_sequence": critical_tex_sequence(source),
                    "translated_sequence": critical_tex_sequence(zh),
                    "source": source, "translated": zh,
                }, ensure_ascii=False))
                raise RuntimeError(f"bare TeX/math skeleton preservation {row['id']}")
            missing_terms = audit_glossary(source, zh)
            if missing_terms:
                raise RuntimeError(f"glossary preservation {row['id']}: {missing_terms}")

            needs_translation = has_english_prose(source)
            if needs_translation and cjk_count(zh) < 2:
                print(json.dumps({"kind":"insufficient_chinese","id":row.get("id"),"source":source,"translated":zh}, ensure_ascii=False))
                raise RuntimeError(f"insufficient Chinese {row['id']}")

            result = dict(row)
            result.update({
                "problem_zh": zh,
                "problem_atomic_original": source,
                "translation_model": args.model if needs_translation else "source-preserved-no-prose",
                "translation_method": "tex-aware-prose-only-glossary-marian-v3" if needs_translation else "source-preserved-no-prose",
                "source_language_before_translation": row.get("language", ""),
                "translation_primary_math_preserved": True,
                "translation_bare_tex_skeleton_preserved": True,
                "translation_math_glossary_preserved": True,
                "translation_has_english_prose_source": needs_translation,
            })
            completed.append(result)

    completed.sort(key=lambda r: int(r["global_atomic_index"]))
    write_jsonl(Path(args.output), completed)
    print(json.dumps({
        "shard": args.shard_id,
        "rows": len(completed),
        "cache_entries": len(cache),
        "translated_rows": sum(bool(r["translation_has_english_prose_source"]) for r in completed),
        "formula_only_rows": sum(not bool(r["translation_has_english_prose_source"]) for r in completed),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
