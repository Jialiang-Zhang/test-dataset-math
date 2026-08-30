#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tex_protection_v3 import primary_math_spans, extract_primary_math

MODEL='facebook/nllb-200-distilled-600M'
CJK_RE=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
NUM_RE=re.compile(r'(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?')
PROTECTED_LITERAL_RE=re.compile(r'(\[ANS\]|\\[$%&#_{}]|(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?)')

SAMPLES=[
    r"Given a hyperbola $\Gamma: \frac{x^2}{a^2} - \frac{y^2}{b^2} = 1$, $F$ is its left focus. The line $y = kx$ intersects the left and right branches of $\Gamma$ at points $A$ and $B$ respectively, satisfying $FA \perp AB$ and $\angle ABF = \angle AFO$ ($O$ is the origin). Find the eccentricity of $\Gamma$.",
    r"Let $1 < p < \infty$. Suppose $\{f_n\}_{n=1}^\infty \subset L^p([0,1])$ are functions such that for each $n \in \mathbb{N}$, $f_n(x) \ge 0$ for a.e.\ $x$. If $f_n$ converges weakly (in $L^p$) to a function $f \in L^p([0,1])$, prove that $f(x) \ge 0$ for a.e.\ $x$.",
    r"A mobile plan charges a base monthly fee of \$15.00 for the first 500 minutes plus \$0.35 for each additional minute. Write a piecewise-defined linear function which calculates the monthly cost $C$ for using $m$ minutes.",
    r"Consider the expression\n\begin{equation}\nI(x)=\int_0^1 t^2 e^{-xt}\,dt.\n\end{equation}\nShow that $I(x)>0$ for $x>0$.",
    r"$3+4$=[ANS]\n$9-3$=[ANS]\n$4 \times 8$=[ANS]\n$15 \div 5$=[ANS]",
]

POST=(('查找','求'),('找出','求'),('显示','证明'),('展示','证明'))


def outside_numbers(text:str):
    chars=list(text)
    for span in primary_math_spans(text):
        for i in range(span.start,span.end): chars[i]=' '
    return Counter(NUM_RE.findall(''.join(chars)))


def has_english(text:str)->bool:
    return bool(re.search(r'[A-Za-z]{2,}', PROTECTED_LITERAL_RE.sub(' ',text)))


def split_gap(gap:str,parts,units):
    cursor=0
    for m in PROTECTED_LITERAL_RE.finditer(gap):
        if m.start()>cursor:
            pre=gap[cursor:m.start()]
            if has_english(pre): parts.append(('trans',len(units)));units.append(pre)
            else: parts.append(('raw',pre))
        parts.append(('raw',m.group(0)));cursor=m.end()
    if cursor<len(gap):
        tail=gap[cursor:]
        if has_english(tail): parts.append(('trans',len(units)));units.append(tail)
        else: parts.append(('raw',tail))


def build(text:str):
    parts=[];units=[];cursor=0
    for span in primary_math_spans(text):
        if span.start>cursor:split_gap(text[cursor:span.start],parts,units)
        parts.append(('raw',text[span.start:span.end]));cursor=span.end
    if cursor<len(text):split_gap(text[cursor:],parts,units)
    return parts,units


def main():
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,src_lang='eng_Latn')
    model=AutoModelForSeq2SeqLM.from_pretrained(MODEL);model.eval()
    target_id=tok.convert_tokens_to_ids('zho_Hans')
    outputs=[]
    for source in SAMPLES:
        parts,units=build(source)
        if units:
            enc=tok(units,return_tensors='pt',padding=True,truncation=True,max_length=512)
            with torch.inference_mode():
                gen=model.generate(**enc,forced_bos_token_id=target_id,num_beams=4,do_sample=False,max_new_tokens=512)
            translated=tok.batch_decode(gen,skip_special_tokens=True)
            translated=[__import__('functools').reduce(lambda s,p:s.replace(*p),POST,x) for x in translated]
        else:translated=[]
        zh=''.join(str(v) if k=='raw' else translated[int(v)] for k,v in parts).strip();outputs.append(zh)
        print('\nSOURCE:',source);print('ZH:',zh)
        assert extract_primary_math(source)==extract_primary_math(zh),(source,zh)
        assert outside_numbers(source)==outside_numbers(zh),(source,zh,outside_numbers(source),outside_numbers(zh))
        if units: assert len(CJK_RE.findall(zh))>=2,(source,zh)
    first=outputs[0]
    assert '焦点' in first,first
    assert ('离心率' in first or '偏心率' in first or '离心' in first or '偏心' in first),first
    weak=outputs[1]
    assert '弱收敛' in weak,weak
    assert '证明' in weak,weak
    cur=outputs[2]
    assert all(x in cur for x in ('15.00','500','0.35')),cur
    assert outputs[-1]==SAMPLES[-1],(SAMPLES[-1],outputs[-1])
    print('\nNLLB protected translation probe passed')

if __name__=='__main__':main()
