#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tex_protection_v3 import extract_primary_math, primary_math_spans

MODEL='Qwen/Qwen3-0.6B'
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
    "任务只有翻译，不是解题。把用户给出的数学题题面翻译为自然、规范的简体中文。"
    "禁止求解、证明、解释、扩写、删减或回答问题。"
    "所有 LaTeX 数学表达式、LaTeX 环境、变量、数字、[ANS]、TeX 转义符必须逐字原样保留。"
    "若输入没有需要翻译的英文自然语言（例如只有公式和 [ANS]），原样输出。"
    "采用标准数学术语：hyperbola=双曲线，focus/foci=焦点，eccentricity=离心率，"
    "converges weakly=弱收敛，holomorphic=全纯，injective=单射。"
    "只输出翻译后的题面，不要加“翻译：”“答案：”“证明：”等前缀。"
)

def outside(text:str)->str:
    chars=list(text)
    for s in primary_math_spans(text):
        for i in range(s.start,s.end):chars[i]=' '
    return ''.join(chars)
def counted(pattern,text):return Counter(pattern.findall(outside(text)))

def translate(src,tok,model):
    msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':src}]
    prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=tok([prompt],return_tensors='pt')
    with torch.inference_mode():
        gen=model.generate(**inputs,max_new_tokens=min(900,max(160,int(inputs.input_ids.shape[1]*1.5))),do_sample=False,repetition_penalty=1.05)
    new=gen[0][inputs.input_ids.shape[1]:]
    return tok.decode(new,skip_special_tokens=True).strip()

def main():
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32);model.eval()
    outs=[]
    for src in SAMPLES:
        zh=translate(src,tok,model);outs.append(zh)
        print('\nSOURCE:',src);print('ZH:',zh)
        assert extract_primary_math(src)==extract_primary_math(zh),(src,zh,extract_primary_math(src),extract_primary_math(zh))
        for pattern,name in ((NUM_RE,'numbers'),(ANS_RE,'ans'),(ESC_RE,'escapes')):
            assert counted(pattern,src)==counted(pattern,zh),(name,src,zh,counted(pattern,src),counted(pattern,zh))
        assert not any(x in zh[:12] for x in ('答案：','证明：','解：','解析：')),(src,zh)
    assert all(x in outs[0] for x in ('双曲线','焦点','离心率')),outs[0]
    assert '弱收敛' in outs[1] and '证明' in outs[1],outs[1]
    assert all(x in outs[2] for x in ('15.00','500','0.35')),outs[2]
    assert '\\begin{equation}' in outs[3] and '\\end{equation}' in outs[3],outs[3]
    assert outs[4]==SAMPLES[4],(SAMPLES[4],outs[4])
    print('\nQwen3 non-thinking translation probe passed')
if __name__=='__main__':main()
