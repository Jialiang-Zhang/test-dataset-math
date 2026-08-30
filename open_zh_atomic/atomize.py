#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

MATH_RE = re.compile(r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$)", re.S)
LABEL_RE = r"(?:[a-hA-H]|\d{1,2}|[ivxIVX]{1,6})"
ZH_LABEL_RE = r"(?:[a-hA-H]|\d{1,2}|[一二三四五六七八九十]{1,4})"
PART_RE = re.compile(
    rf"(?im)(?P<marker>\bPart\s*(?:\({LABEL_RE}\)|{LABEL_RE}\s*[\).:：])|\({LABEL_RE}\)|（{ZH_LABEL_RE}）|^[ \t]*{LABEL_RE}\s*[\).:：])"
)
EN_TASK_WORDS = (
    "prove|show|find|determine|compute|calculate|evaluate|solve|classify|construct|describe|verify|"
    "establish|derive|give|decide|characterize|state|write|express|graph|sketch|draw|plot|use|estimate|"
    "simplify|factor|expand|differentiate|integrate|convert|identify|explain|compare|interpret|list|name|"
    "complete|select|choose|tabulate|record"
)
ZH_TASK_WORDS = (
    "证明出|证明|求证|求极限|求值|求出|求解|计算|判断|说明|确定|找出|解答|构造|给出|分类|刻画|"
    "验证|推导|导出|写出|表示|作图|画出|绘制|使用|利用|估计|化简|因式分解|展开|求导|微分|积分|"
    "转换|识别|解释|比较|列出|命名|完成|选择|制表|记录|求|解"
)
EN_TASK_RE = re.compile(rf"(?i)\b({EN_TASK_WORDS})\b")
ZH_TASK_RE = re.compile(rf"({ZH_TASK_WORDS})")
DIRECTIVE_RE = re.compile(
    r"(?i)(answer\s+the\s+following|solve\s+the\s+following|prove\s+the\s+following|show\s+the\s+following|"
    r"find\s+the\s+following|determine\s+the\s+following|complete\s+the\s+following|for\s+each\s+(?:of\s+the\s+following|problem)|"
    r"in\s+each\s+part|回答下列|解答下列|求下列|证明下列|完成下列|以下各题|各小题|分别回答|分别求|分别证明)"
)
QUESTION_LEAD_RE = re.compile(
    r"(?i)\b(?:what|which|who|whom|whose|where|when|why|how|is|are|was|were|do|does|did|can|could|will|would|should|must|may)\b"
    r"|(?:什么|哪个|哪些|多少|如何|是否|为何|为什么|哪一个|哪一种)"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?;；])\s+|(?<=\.)\s+(?=[A-Z])|\n{2,}")
CONJOINED_TASK_RE = re.compile(
    rf"(?is)(?:\s*[,;，；]\s*|\s+\b(?:and|then|also|moreover|finally)\b\s+|\s*(?:并且|并|且|再|另外|此外|然后|最后|进而)\s*)"
    rf"(?P<task>\b(?:{EN_TASK_WORDS})\b|(?:{ZH_TASK_WORDS}))"
)


def mask_math(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in m.group(0))
    return MATH_RE.sub(repl, text)


def first_task_match(masked: str) -> re.Match[str] | None:
    en, zh = EN_TASK_RE.search(masked), ZH_TASK_RE.search(masked)
    if en is None: return zh
    if zh is None: return en
    return en if en.start() <= zh.start() else zh


def first_request_match(masked: str) -> re.Match[str] | None:
    task, question = first_task_match(masked), QUESTION_LEAD_RE.search(masked)
    if task is None: return question
    if question is None: return task
    return task if task.start() <= question.start() else question


def has_task(text: str) -> bool:
    return first_task_match(mask_math(text)) is not None


def normalize_label(label: str) -> tuple[str, int] | None:
    inner = re.sub(r"(?i)^part\s*", "", label.strip()).strip("()（）. :：\t")
    if inner.isdigit(): return ("num", int(inner))
    low = inner.casefold()
    if len(low) == 1 and "a" <= low <= "h": return ("alpha", ord(low) - 96)
    roman = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10}
    if low in roman: return ("roman", roman[low])
    zh = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
    if inner in zh: return ("zhnum", zh[inner])
    return None


def canonical_label(label: str) -> str:
    inner = re.sub(r"(?i)^part\s*", "", label.strip()).strip("()（）. :：\t")
    return inner or "q"


def sequential_markers(matches: list[re.Match[str]]) -> bool:
    labels = [normalize_label(m.group("marker")) for m in matches]
    if any(x is None for x in labels) or len(labels) < 2: return False
    vals = [x for x in labels if x is not None]
    if len({x[0] for x in vals}) != 1: return False
    return all(vals[i+1][1] == vals[i][1] + 1 for i in range(len(vals)-1))


def split_explicit_parts(text: str) -> list[tuple[str, str]] | None:
    masked = mask_math(text)
    matches = list(PART_RE.finditer(masked))
    if len(matches) < 2 or not sequential_markers(matches): return None
    prefix = text[:matches[0].start()].strip()
    bodies: list[tuple[str,str]] = []
    task_bodies = ans_bodies = question_bodies = 0
    marker_texts: list[str] = []
    for i, m in enumerate(matches):
        body = text[m.end():(matches[i+1].start() if i+1 < len(matches) else len(text))].strip()
        if len(body) >= 4 and has_task(body): task_bodies += 1
        if "[ANS]" in body: ans_bodies += 1
        if re.search(r"[?？]", mask_math(body)): question_bodies += 1
        marker_texts.append(m.group("marker").strip())
        bodies.append((m.group("marker"), body))
    directive = bool(DIRECTIVE_RE.search(mask_math(prefix)))
    answer_template = ans_bodies >= 2
    explicit_part_style = all(re.match(r"(?i)^\s*Part\b", marker) for marker in marker_texts)
    if not directive and task_bodies < 2 and not answer_template and not (explicit_part_style and question_bodies >= 2):
        return None
    out = []
    for label, body in bodies:
        if len(body) < 3: continue
        out.append((label, f"{prefix}\n{body}".strip() if prefix else body))
    return out if len(out) >= 2 else None


def split_conjoined_tasks(text: str) -> list[tuple[str, str]] | None:
    masked = mask_math(text)
    first = first_request_match(masked)
    if first is None: return None
    later = [m for m in CONJOINED_TASK_RE.finditer(masked, first.end()) if m.start("task") > first.start()]
    if not later: return None
    shared = text[:first.start()].strip(" ,，:：")
    starts = [first.start()] + [m.start("task") for m in later]
    ends = [m.start() for m in later] + [len(text)]
    out = []
    for i, (start, end) in enumerate(zip(starts, ends), 1):
        clause = text[start:end].strip(" ,，;；")
        if len(clause) < 4 or first_request_match(mask_math(clause)) is None: continue
        out.append((f"task{i}", f"{shared} {clause}".strip() if shared else clause))
    return out if len(out) >= 2 else None


def split_task_sentences(text: str) -> list[tuple[str, str]] | None:
    masked = mask_math(text)
    spans, last = [], 0
    for m in SENTENCE_SPLIT_RE.finditer(masked):
        spans.append((last, m.start())); last = m.end()
    spans.append((last, len(text)))
    parts = [(a,b,text[a:b].strip(),masked[a:b].strip()) for a,b in spans if text[a:b].strip()]
    task_idxs = [i for i,p in enumerate(parts) if first_task_match(p[3]) is not None]
    if len(task_idxs) < 2: return None
    for i in task_idxs[1:]:
        ms = parts[i][3].lstrip()
        if not (EN_TASK_RE.match(ms) or ZH_TASK_RE.match(ms) or re.match(r"(?i)^(hence|then|also|moreover|finally)\s+", ms) or re.match(r"^(再|并且|另外|此外|最后|进而|然后)", ms)):
            return None
    first_i = task_idxs[0]
    first_raw, first_mask = parts[first_i][2], parts[first_i][3]
    fm = first_task_match(first_mask)
    if fm is None: return None
    shared = " ".join(p[2] for p in parts[:first_i]).strip()
    local_prefix = first_raw[:fm.start()].strip(" ,，:：")
    if local_prefix: shared = (shared + " " + local_prefix).strip()
    out = []
    for j, i in enumerate(task_idxs, 1):
        raw, ms = parts[i][2], parts[i][3]
        match = first_task_match(ms)
        if match is None: continue
        clause = raw[match.start():].strip()
        child = f"{shared} {clause}".strip() if shared else clause
        if len(child) >= 4: out.append((f"task{j}", child))
    return out if len(out) >= 2 else None


def split_question_marks(text: str) -> list[tuple[str, str]] | None:
    masked = mask_math(text)
    qpos = [m.start() for m in re.finditer(r"[?？]", masked)]
    if len(qpos) < 2: return None
    chunks, start = [], 0
    for pos in qpos:
        chunk = text[start:pos+1].strip()
        if chunk: chunks.append(chunk)
        start = pos + 1
    tail = text[start:].strip()
    if tail: chunks.append(tail)
    valid = [c for c in chunks if has_task(c) or c.rstrip().endswith(("?","？"))]
    if len(valid) < 2: return None
    first_mask = mask_math(valid[0])
    request = first_request_match(first_mask)
    shared = valid[0][:request.start()].strip(" ,，:：") if request else ""
    out = []
    for i,c in enumerate(valid,1):
        child = c
        if i > 1 and shared and not child.startswith(shared): child = f"{shared} {child}".strip()
        out.append((f"question{i}", child))
    return out if len(out) >= 2 else None


def split_atomic_problem(text: str) -> tuple[list[tuple[str,str]], str]:
    for reason, fn in (("explicit_parts",split_explicit_parts),("conjoined_tasks",split_conjoined_tasks),("separate_task_sentences",split_task_sentences),("multiple_question_sentences",split_question_marks)):
        parts = fn(text)
        if parts: return parts, reason
    return [("q1", text.strip())], "unsplit"


def fully_split(text: str, max_depth: int = 8) -> tuple[list[dict[str,Any]], Counter[str]]:
    leaves: list[dict[str,Any]] = []
    reasons: Counter[str] = Counter()
    def visit(current: str, labels: list[str], chain: list[str], depth: int) -> None:
        parts, reason = split_atomic_problem(current)
        if len(parts) <= 1:
            leaves.append({"text": current.strip(), "label_path": labels[:] if labels else ["q1"], "reason_chain": chain[:]})
            return
        if depth >= max_depth: raise RuntimeError(f"split depth exceeded: {current[:400]}")
        reasons[reason] += 1
        for label, child in parts:
            child = child.strip()
            if not child or child == current.strip(): raise RuntimeError(f"non-progress split: {reason}")
            visit(child, labels + [canonical_label(label)], chain + [reason], depth + 1)
    visit(text.strip(), [], [], 0)
    for leaf in leaves:
        check, reason = split_atomic_problem(leaf["text"])
        if len(check) > 1: raise RuntimeError(f"residual multiquestion ({reason}): {leaf['text'][:400]}")
    return leaves, reasons


def load_jsonl(path: Path) -> list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="open_zh_atomic/work/raw/all_open_p3_exact_unique.jsonl")
    ap.add_argument("--output", default="open_zh_atomic/work/prepared.jsonl")
    ap.add_argument("--report", default="open_zh_atomic/work/atomization_report.json")
    args = ap.parse_args()
    source = load_jsonl(Path(args.input))
    atomic: list[dict[str,Any]] = []
    split_parents = detached = max_depth = max_parts = 0
    node_reasons: Counter[str] = Counter()
    for source_idx,row in enumerate(source):
        problem = str(row.get("problem") or "").strip()
        if not problem: raise RuntimeError(f"empty source problem {source_idx}")
        leaves,reasons = fully_split(problem)
        node_reasons.update(reasons)
        is_split = len(leaves) > 1
        if is_split: split_parents += 1
        max_parts = max(max_parts, len(leaves))
        parent_id = str(row["parent_id"])
        for atomic_index,leaf in enumerate(leaves,1):
            out = dict(row)
            chain = [str(x) for x in leaf["reason_chain"]]
            labels = [str(x) for x in leaf["label_path"]]
            max_depth = max(max_depth,len(chain))
            if is_split:
                out["parent_answer"] = out.get("answer","")
                out["parent_solution"] = out.get("solution","")
                out["answer"] = ""; out["solution"] = ""; detached += 1
            out.update({
                "id": f"{parent_id}::q{atomic_index}", "parent_id": parent_id,
                "parent_problem": problem, "atomic_source_problem": leaf["text"],
                "atomic_label": ".".join(labels), "atomic_label_path": labels,
                "atomic_index": atomic_index, "atomic_count_from_parent": len(leaves),
                "split_reason": chain[0] if chain else "unsplit", "split_reason_chain": chain,
                "split_depth": len(chain), "source_row_index": source_idx,
                "global_atomic_index": len(atomic),
                "answer_scope": "parent_aggregate_not_attached" if is_split else "atomic_or_parent",
            })
            atomic.append(out)
    report = {
        "source_rows":len(source), "atomic_rows_pretranslation":len(atomic), "split_parent_rows":split_parents,
        "unsplit_parent_rows":len(source)-split_parents, "coverage_source_rows":len({r['source_row_index'] for r in atomic}),
        "split_child_rows_with_parent_answer_detached":detached, "max_split_depth":max_depth,
        "max_atomic_parts_from_parent":max_parts, "recursive_split_node_reasons":dict(node_reasons),
        "residual_multiquestion_rows":0,
    }
    if report["coverage_source_rows"] != len(source): raise RuntimeError(report)
    write_jsonl(Path(args.output),atomic)
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
