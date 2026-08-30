#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
import torch
from transformers import MarianMTModel, MarianTokenizer

from tex_protection_v3 import primary_math_spans, extract_primary_math

MODEL='Helsinki-NLP/opus-mt-en-zh'
CJK_RE=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
NUM_RE=re.compile(r'(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?')
ANS_RE=re.compile(r'\[ANS\]')

SAMPLES=[
    r"Given a hyperbola $\Gamma: \frac{x^2}{a^2} - \frac{y^2}{b^2} = 1$, $F$ is its left focus. The line $y = kx$ intersects the left and right branches of $\Gamma$ at points $A$ and $B$ respectively, satisfying $FA \perp AB$ and $\angle ABF = \angle AFO$ ($O$ is the origin). Find the eccentricity of $\Gamma$.",
    r"Let $1 < p < \infty$. Suppose $\{f_n\}_{n=1}^\infty \subset L^p([0,1])$ are functions such that for each $n \in \mathbb{N}$, $f_n(x) \ge 0$ for a.e.\ $x$. If $f_n$ converges weakly (in $L^p$) to a function $f \in L^p([0,1])$, prove that $f(x) \ge 0$ for a.e.\ $x$.",
    r"A mobile plan charges a base monthly fee of \$15.00 for the first 500 minutes plus \$0.35 for each additional minute. Write a piecewise-defined linear function which calculates the monthly cost $C$ for using $m$ minutes.",
    r"Consider the expression\n\begin{equation}\nI(x)=\int_0^1 t^2 e^{-xt}\,dt.\n\end{equation}\nShow that $I(x)>0$ for $x>0$.",
    r"$3+4$=[ANS]\n$9-3$=[ANS]\n$4 \times 8$=[ANS]\n$15 \div 5$=[ANS]",
]

POST_REPLACEMENTS=(
    ('查找', '求'),
    ('找出', '求'),
    ('展示', '证明'),
    ('显示', '证明'),
)


def preprocess_gap(gap:str)->str:
    # Keep the MT input English-only. Normalize only TeX prose formatting.
    return gap.replace(r'\$', ' USD ').replace(r'\%', ' percent ').replace(r'\ ', ' ')


def postprocess_zh(text:str)->str:
    out=text
    for old,new in POST_REPLACEMENTS:
        out=out.replace(old,new)
    return out


def translatable(gap:str)->bool:
    stripped=ANS_RE.sub(' ',gap)
    return bool(re.search(r'[A-Za-z]',stripped))


def build(text:str):
    parts=[];units=[];cursor=0
    def append_gap(gap:str):
        last=0
        for m in ANS_RE.finditer(gap):
            pre=gap[last:m.start()]
            if pre:
                if translatable(pre):parts.append(('trans',len(units)));units.append(pre)
                else:parts.append(('raw',pre))
            parts.append(('raw',m.group(0)));last=m.end()
        tail=gap[last:]
        if tail:
            if translatable(tail):parts.append(('trans',len(units)));units.append(tail)
            else:parts.append(('raw',tail))
    for span in primary_math_spans(text):
        if span.start>cursor:append_gap(text[cursor:span.start])
        parts.append(('raw',text[span.start:span.end]));cursor=span.end
    if cursor<len(text):append_gap(text[cursor:])
    return parts,units


def numbers_outside_math(text:str):
    chars=list(text)
    for span in primary_math_spans(text):
        for i in range(span.start,span.end):chars[i]=' '
    return Counter(NUM_RE.findall(''.join(chars)))


def translate_one(source,tok,model):
    parts,units=build(source)
    mt=[preprocess_gap(x) for x in units]
    if mt:
        enc=tok(mt,return_tensors='pt',padding=True,truncation=True,max_length=512)
        with torch.inference_mode():gen=model.generate(**enc,num_beams=4,do_sample=False,max_new_tokens=512,renormalize_logits=True)
        out_units=[postprocess_zh(x) for x in tok.batch_decode(gen,skip_special_tokens=True)]
    else:out_units=[]
    return ''.join(str(v) if k=='raw' else out_units[int(v)] for k,v in parts).strip(),units


def main():
    torch.set_num_threads(2)
    tok=MarianTokenizer.from_pretrained(MODEL)
    model=MarianMTModel.from_pretrained(MODEL);model.eval()
    outputs=[]
    for source in SAMPLES:
        zh,units=translate_one(source,tok,model);outputs.append(zh)
        print('\nSOURCE:',source);print('ZH:',zh)
        assert extract_primary_math(source)==extract_primary_math(zh),(source,zh)
        assert numbers_outside_math(source)==numbers_outside_math(zh),(source,zh,numbers_outside_math(source),numbers_outside_math(zh))
        if units:assert len(CJK_RE.findall(zh))>=2,(source,zh)
    first=outputs[0]
    assert '焦点' in first,first
    assert ('偏心' in first or '离心' in first),first
    assert '查找' not in first,first
    weak=outputs[1]
    assert '弱收敛' in weak,weak
    assert '证明' in weak,weak
    currency=outputs[2]
    assert all(x in currency for x in ('15.00','500','0.35')),currency
    # Pure formula row stays byte-identical.
    assert outputs[-1]==SAMPLES[-1],(SAMPLES[-1],outputs[-1])
    print('\ncoarse monolingual translation probe passed')

if __name__=='__main__':main()
