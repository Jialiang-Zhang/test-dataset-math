#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tex_protection_v3 import extract_primary_math, primary_math_spans

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
NUM_RE=re.compile(r'(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?')
ANS_RE=re.compile(r'\[ANS\]')
ESC_RE=re.compile(r'\\[$%&#_{}]')

SAMPLES=[
    r"Given a hyperbola $\Gamma: \frac{x^2}{a^2} - \frac{y^2}{b^2} = 1$, $F$ is its left focus. The line $y = kx$ intersects the left and right branches of $\Gamma$ at points $A$ and $B$ respectively, satisfying $FA \perp AB$ and $\angle ABF = \angle AFO$ ($O$ is the origin). Find the eccentricity of $\Gamma$.",
    r"Let $1 < p < \infty$. Suppose $\{f_n\}_{n=1}^\infty \subset L^p([0,1])$ are functions such that for each $n \in \mathbb{N}$, $f_n(x) \ge 0$ for a.e.\ $x$. If $f_n$ converges weakly (in $L^p$) to a function $f \in L^p([0,1])$, prove that $f(x) \ge 0$ for a.e.\ $x$.",
    r"A mobile plan charges a base monthly fee of \$15.00 for the first 500 minutes plus \$0.35 for each additional minute. Write a piecewise-defined linear function which calculates the monthly cost $C$ for using $m$ minutes.",
    r"Consider the expression\n\begin{equation}\nI(x)=\int_0^1 t^2 e^{-xt}\,dt.\n\end{equation}\nShow that $I(x)>0$ for $x>0$.",
    r"$3+4$=[ANS]\n$9-3$=[ANS]\n$4 \times 8$=[ANS]\n$15 \div 5$=[ANS]",
]

SYSTEM=(
    "你是专业数学竞赛题翻译器。把英文题面翻译成自然、规范、简洁的简体中文。"
    "只翻译自然语言，不求解、不解释、不补充。严格原样保留所有 LaTeX 数学内容、数学命令、变量、数字、[ANS]、TeX 转义符和换行结构。"
    "术语必须采用标准数学中文，例如 hyperbola=双曲线，focus/foci=焦点，eccentricity=离心率，"
    "weakly converges=弱收敛，holomorphic=全纯，injective=单射。只输出翻译后的题面。"
)


def outside(text:str)->str:
    chars=list(text)
    for s in primary_math_spans(text):
        for i in range(s.start,s.end):chars[i]=' '
    return ''.join(chars)

def counted(pattern,text):return Counter(pattern.findall(outside(text)))

def translate(source,tok,model):
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':source}]
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=tok([text],return_tensors='pt')
    with torch.inference_mode():
        generated=model.generate(**inputs,max_new_tokens=1200,do_sample=False,repetition_penalty=1.02)
    new=generated[0][inputs.input_ids.shape[1]:]
    return tok.decode(new,skip_special_tokens=True).strip()


def main():
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);model.eval()
    outs=[]
    for src in SAMPLES:
        zh=translate(src,tok,model);outs.append(zh)
        print('\nSOURCE:',src);print('ZH:',zh)
        assert extract_primary_math(src)==extract_primary_math(zh),(src,zh,extract_primary_math(src),extract_primary_math(zh))
        for pattern,name in ((NUM_RE,'numbers'),(ANS_RE,'ans'),(ESC_RE,'escapes')):
            assert counted(pattern,src)==counted(pattern,zh),(name,src,zh,counted(pattern,src),counted(pattern,zh))
    assert '双曲线' in outs[0] and '焦点' in outs[0] and '离心率' in outs[0],outs[0]
    assert '弱收敛' in outs[1] and '证明' in outs[1],outs[1]
    assert all(x in outs[2] for x in ('15.00','500','0.35')),outs[2]
    assert '\\begin{equation}' in outs[3] and '\\end{equation}' in outs[3],outs[3]
    # No English prose exists in this row, so the safest valid translation is exact preservation.
    assert outs[4]==SAMPLES[4],(SAMPLES[4],outs[4])
    print('\nQwen full-context translation probe passed')

if __name__=='__main__':main()
