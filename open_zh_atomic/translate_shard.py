#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import MarianMTModel, MarianTokenizer
from glossary import PATTERN as GLOSSARY_RE, rule_for, audit as audit_glossary

MATH_RE = re.compile(r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$)",re.S)
NUMBER_TOKEN = r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?"
ESCAPE_TOKEN = r"\\[$%&#_{}]"
PLACEHOLDER_TOKEN = r"\[ANS\]"
NUMBER_RE = re.compile(NUMBER_TOKEN)
ESCAPE_RE = re.compile(ESCAPE_TOKEN)
PLACEHOLDER_RE = re.compile(PLACEHOLDER_TOKEN)
PROTECTED_RE = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$|"+ESCAPE_TOKEN+r"|"+PLACEHOLDER_TOKEN+r"|"+NUMBER_TOKEN+r")",re.S
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def load_jsonl(path: Path) -> list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_jsonl(path: Path,rows:list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")

def cjk(text:str)->int: return len(CJK_RE.findall(text))
def latin(text:str)->int: return len(LATIN_RE.findall(text))

def add_translatable(text:str,parts:list[tuple[str,str|int]],units:list[str]) -> None:
    if not text: return
    if latin(text) >= 1:
        idx=len(units); units.append(text); parts.append(("trans",idx))
    else:
        parts.append(("raw",text))

def append_plain(plain:str,parts:list[tuple[str,str|int]],units:list[str]) -> None:
    cursor=0
    for m in GLOSSARY_RE.finditer(plain):
        if m.start()>cursor: add_translatable(plain[cursor:m.start()],parts,units)
        parts.append(("raw",rule_for(m).zh))
        cursor=m.end()
    if cursor<len(plain): add_translatable(plain[cursor:],parts,units)

def structure(text:str) -> tuple[list[tuple[str,str|int]],list[str]]:
    parts:list[tuple[str,str|int]]=[]; units:list[str]=[]; cursor=0
    for m in PROTECTED_RE.finditer(text):
        if m.start()>cursor: append_plain(text[cursor:m.start()],parts,units)
        parts.append(("raw",m.group(0))); cursor=m.end()
    if cursor<len(text): append_plain(text[cursor:],parts,units)
    return parts,units

def sanitize(src:str,dst:str)->str:
    out=dst.replace(r"\$","").replace("$","")
    out=out.replace(r"\[","").replace(r"\]","").replace(r"\(","").replace(r"\)","").replace("[ANS]","")
    out=re.sub(r"\\(?=[$%&#_{}])","",out)
    if "?" not in src and "？" not in src: out=out.replace("?","").replace("？","")
    if "!" not in src and "！" not in src: out=out.replace("!","").replace("！","")
    return out.strip()

def translate_units(units:list[str],tok:MarianTokenizer,model:MarianMTModel,batch_size:int,cache:dict[str,str])->list[str]:
    missing=[]; seen=set()
    for u in units:
        if u not in cache and u not in seen: seen.add(u); missing.append(u)
    for start in range(0,len(missing),batch_size):
        batch=missing[start:start+batch_size]
        enc=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=512)
        with torch.inference_mode(): gen=model.generate(**enc,num_beams=1,do_sample=False,max_new_tokens=512)
        dec=tok.batch_decode(gen,skip_special_tokens=True)
        for src,dst in zip(batch,dec): cache[src]=sanitize(src,dst)
    return [cache[u] for u in units]

def reassemble(parts:list[tuple[str,str|int]],translated:list[str])->str:
    return "".join(str(v) if k=="raw" else translated[int(v)] for k,v in parts).strip()

def math_spans(text:str)->list[str]: return MATH_RE.findall(text)
def outside_math(text:str)->str: return MATH_RE.sub(" ",text)
def literals(pattern:re.Pattern[str],text:str)->Counter[str]: return Counter(pattern.findall(outside_math(text)))
def preserved(counter:Counter[str],translated:str)->bool: return all(translated.count(x)>=n for x,n in counter.items())


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="open_zh_atomic/work/prepared.jsonl")
    ap.add_argument("--output",required=True)
    ap.add_argument("--shard-id",type=int,required=True)
    ap.add_argument("--shard-count",type=int,default=32)
    ap.add_argument("--batch-size",type=int,default=24)
    ap.add_argument("--model",default="Helsinki-NLP/opus-mt-en-zh")
    args=ap.parse_args()
    torch.set_num_threads(max(1,int(os.environ.get("TORCH_NUM_THREADS","2"))))
    rows=load_jsonl(Path(args.input))
    selected=[r for r in rows if int(r["global_atomic_index"])%args.shard_count==args.shard_id]
    if not selected: raise RuntimeError(f"empty shard {args.shard_id}")
    tok=MarianTokenizer.from_pretrained(args.model); model=MarianMTModel.from_pretrained(args.model); model.eval()
    cache:dict[str,str]={}; out_rows=[]
    for row in selected:
        source=str(row["atomic_source_problem"])
        lang=str(row.get("language") or "").lower()
        if lang.startswith("zh"):
            zh=source; method="source-preserved"
        else:
            parts,units=structure(source)
            trans=translate_units(units,tok,model,args.batch_size,cache)
            zh=reassemble(parts,trans); method="math-literal-glossary-protected-marian"
        if not zh: raise RuntimeError(f"empty translation {row['id']}")
        if math_spans(source)!=math_spans(zh): raise RuntimeError(f"math preservation {row['id']}")
        for pattern,name in ((NUMBER_RE,"numbers"),(ESCAPE_RE,"escapes"),(PLACEHOLDER_RE,"placeholders")):
            if not preserved(literals(pattern,source),zh): raise RuntimeError(f"{name} preservation {row['id']}")
        missing=audit_glossary(outside_math(source),zh)
        if missing: raise RuntimeError(f"glossary preservation {row['id']}: {missing}")
        if lang.startswith("en") and latin(outside_math(source))>=4 and cjk(outside_math(zh))<2:
            raise RuntimeError(f"insufficient Chinese {row['id']}")
        result=dict(row)
        result.update({
            "problem_zh":zh,"problem_atomic_original":source,"translation_model":args.model if method!="source-preserved" else "source-preserved",
            "translation_method":method,"source_language_before_translation":row.get("language",""),
            "translation_math_spans_preserved":True,"translation_nonmath_numbers_preserved":True,
            "translation_escaped_literals_preserved":True,"translation_placeholders_preserved":True,
            "translation_math_glossary_preserved":True,
        })
        out_rows.append(result)
    out_rows.sort(key=lambda r:int(r["global_atomic_index"]))
    write_jsonl(Path(args.output),out_rows)
    print(json.dumps({"shard":args.shard_id,"rows":len(out_rows),"cache_entries":len(cache)},ensure_ascii=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
