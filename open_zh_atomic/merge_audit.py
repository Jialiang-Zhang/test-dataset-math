#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from atomize import MATH_RE, split_atomic_problem
from glossary import audit as audit_glossary

CJK_RE=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE=re.compile(r"[A-Za-z]")
NUMBER_RE=re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?")
ESCAPE_RE=re.compile(r"\\[$%&#_{}]")
PLACEHOLDER_RE=re.compile(r"\[ANS\]")

def load_jsonl(path:Path)->list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def write_jsonl(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")
def plain(text:str)->str:return MATH_RE.sub(" ",text)
def cjk(text:str)->int:return len(CJK_RE.findall(text))
def latin(text:str)->int:return len(LATIN_RE.findall(text))
def lit(pattern:re.Pattern[str],text:str)->Counter[str]:return Counter(pattern.findall(plain(text)))
def preserved(c:Counter[str],text:str)->bool:return all(text.count(x)>=n for x,n in c.items())
def norm_hash(text:str)->str:return hashlib.sha256(re.sub(r"\s+","",text.casefold()).encode()).hexdigest()

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepared",default="open_zh_atomic/work/prepared.jsonl")
    ap.add_argument("--shards-dir",default="open_zh_atomic/work/shards")
    ap.add_argument("--output-dir",default="open_zh_atomic/output")
    args=ap.parse_args()
    prepared=load_jsonl(Path(args.prepared))
    expected={int(r["global_atomic_index"]):r for r in prepared}
    if len(expected)!=len(prepared):raise RuntimeError("duplicate prepared index")
    translated=[]
    files=sorted(Path(args.shards_dir).rglob("*.jsonl"))
    if not files:raise RuntimeError("no shard files")
    for f in files:translated.extend(load_jsonl(f))
    got={}
    for r in translated:
        idx=int(r["global_atomic_index"])
        if idx in got:raise RuntimeError(f"duplicate translated index {idx}")
        got[idx]=r
    if set(got)!=set(expected):
        missing=sorted(set(expected)-set(got));extra=sorted(set(got)-set(expected))
        raise RuntimeError(f"coverage mismatch missing={missing[:20]} extra={extra[:20]}")
    rows=[got[i] for i in sorted(got)]
    issues=[];by_source=Counter();by_language=Counter();methods=Counter();parents=set();split_children=0
    for r in rows:
        idx=int(r["global_atomic_index"]); e=expected[idx]
        zh=str(r.get("problem_zh") or "").strip(); original=str(r.get("problem_atomic_original") or "")
        row_issues=[]
        if r.get("id")!=e.get("id"):row_issues.append("id_mismatch")
        if r.get("parent_id")!=e.get("parent_id"):row_issues.append("parent_id_mismatch")
        if original!=e.get("atomic_source_problem"):row_issues.append("atomic_source_mismatch")
        if not zh:row_issues.append("empty_problem")
        else:
            parts,reason=split_atomic_problem(zh)
            if len(parts)>1:row_issues.append(f"residual_multiquestion:{reason}:{len(parts)}")
            if MATH_RE.findall(original)!=MATH_RE.findall(zh):row_issues.append("math_not_preserved")
            for pattern,name in ((NUMBER_RE,"numbers"),(ESCAPE_RE,"escapes"),(PLACEHOLDER_RE,"placeholders")):
                if not preserved(lit(pattern,original),zh):row_issues.append(f"{name}_not_preserved")
            miss=audit_glossary(plain(original),zh)
            if miss:row_issues.append("glossary_missing:"+",".join(miss))
            lang=str(r.get("source_language_before_translation") or e.get("language") or "")
            if lang.startswith("en") and latin(plain(original))>=4 and cjk(plain(zh))<2:row_issues.append("insufficient_chinese")
        if int(e.get("atomic_count_from_parent") or 1)>1:
            split_children+=1
            if e.get("answer_scope")!="parent_aggregate_not_attached":row_issues.append("bad_answer_scope")
            if e.get("answer") or e.get("solution") or r.get("answer") or r.get("solution"):row_issues.append("split_child_inherited_parent_answer")
        if not r.get("translation_math_spans_preserved"):row_issues.append("worker_math_flag_false")
        if not r.get("translation_math_glossary_preserved"):row_issues.append("worker_glossary_flag_false")
        if row_issues:issues.append({"id":r.get("id"),"idx":idx,"issues":row_issues,"problem":zh,"original":original})
        parents.add(str(r.get("parent_id")));by_source[str(r.get("source_family"))]+=1;by_language[str(r.get("source_language_before_translation"))]+=1;methods[str(r.get("translation_method"))]+=1
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/"needs_review.jsonl",issues)
    report={
        "prepared_atomic_rows":len(prepared),"translated_atomic_rows":len(rows),"unique_parent_rows_covered":len(parents),
        "issue_rows":len(issues),"residual_multiquestion_rows":sum(any(x.startswith("residual_multiquestion") for x in i["issues"]) for i in issues),
        "split_children_answer_detachment_audited":split_children,"by_source_family":dict(by_source),"by_original_language":dict(by_language),
        "translation_methods":dict(methods),"unique_chinese_problem_hashes":len({norm_hash(str(r.get('problem_zh') or '')) for r in rows}),
        "audit_contract":{"one_question_per_row":True,"recursive_atomization":True,"math_spans_preserved":True,"protected_literals_preserved":True,"math_glossary_preserved":True,"english_prose_is_chinese":True,"split_children_do_not_inherit_parent_answers":True}
    }
    (out/"audit_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if issues:
        print(json.dumps(report,ensure_ascii=False));print(json.dumps(issues[:20],ensure_ascii=False,indent=2));raise RuntimeError(f"audit failed: {len(issues)} rows")
    final=[]
    for idx,r in enumerate(rows):
        final.append({
            "idx":idx,"problem":r["problem_zh"],"problem_original":r["problem_atomic_original"],"subject":r.get("subject",""),
            "source":r.get("source",""),"source_family":r.get("source_family",""),"year":r.get("year"),"difficulty":r.get("difficulty",""),
            "parent_id":r.get("parent_id",""),"normalized_parent_sha256":r.get("normalized_sha256",""),"atomic_index":r.get("atomic_index",1),
            "atomic_count_from_parent":r.get("atomic_count_from_parent",1),"atomic_label_path":r.get("atomic_label_path",[]),"split_reason_chain":r.get("split_reason_chain",[]),
            "source_language":r.get("source_language_before_translation",""),"provenance_url":r.get("provenance_url",""),"provenance_tier":"P3","license":r.get("license",""),
            "atomicity_status":"passed","translation_status":"passed"
        })
    write_jsonl(out/"questions_zh_atomic_p3.jsonl",final)
    minimal=[{"idx":r["idx"],"problem":r["problem"],"subject":r["subject"],"source_family":r["source_family"],"parent_id":r["parent_id"],"normalized_parent_sha256":r["normalized_parent_sha256"]} for r in final]
    write_jsonl(out/"questions_zh_atomic_p3_min.jsonl",minimal)
    (out/"README.md").write_text(f"# Open P3 中文原子数学题库\n\n- 父题覆盖：{len(parents)}\n- 中文原子题：{len(final)}\n- 审计问题：0\n- 每条 JSONL 仅一个明确问题；多问递归拆分。\n- 仅包含明确开源 P3 来源：OlymMATH、HARDMath、MathOdyssey、MA-ProofBench、UGMathBench。\n- 数学公式、数字、TeX 转义和 [ANS] 占位符受保护；高置信数学术语使用受控词汇表。\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
    return 0

if __name__=="__main__":raise SystemExit(main())
