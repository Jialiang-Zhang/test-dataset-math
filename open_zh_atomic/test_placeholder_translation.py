#!/usr/bin/env python3
from __future__ import annotations

import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL='Helsinki-NLP/opus-mt-en-zh'
SAMPLES=[
    ('Given 公式甲, 公式乙 is its left focus. The line 公式丙 intersects the two branches. Find the eccentricity.', ['公式甲','公式乙','公式丙']),
    ('A mobile plan charges 数值甲 dollars for the first 数值乙 minutes plus 数值丙 dollars for each additional minute. Write the cost 公式甲 as a function of 公式乙.', ['数值甲','数值乙','数值丙','公式甲','公式乙']),
    ('Suppose 公式甲 are functions such that 公式乙. If 公式丙 converges weakly to 公式丁, prove that 公式戊.', ['公式甲','公式乙','公式丙','公式丁','公式戊']),
    ('Let 术语甲 be an 术语乙 术语丙 and determine its image.', ['术语甲','术语乙','术语丙']),
    ('For 公式甲, compute 公式乙; then use 公式丙 to prove 公式丁.', ['公式甲','公式乙','公式丙','公式丁']),
]


def main():
    torch.set_num_threads(2)
    tok=MarianTokenizer.from_pretrained(MODEL)
    model=MarianMTModel.from_pretrained(MODEL); model.eval()
    srcs=[x[0] for x in SAMPLES]
    enc=tok(srcs,return_tensors='pt',padding=True,truncation=True,max_length=512)
    with torch.inference_mode():
        gen=model.generate(**enc,num_beams=4,do_sample=False,max_new_tokens=512,renormalize_logits=True)
    outs=tok.batch_decode(gen,skip_special_tokens=True)
    for (src,markers),dst in zip(SAMPLES,outs):
        print('SRC:',src); print('DST:',dst); print('MARKERS',markers); print()
        assert all(dst.count(m)==1 for m in markers),(src,dst,markers)
    assert '焦点' in outs[0],outs[0]
    assert ('偏心' in outs[0] or '离心' in outs[0]),outs[0]
    print('natural Chinese entity-label placeholder probe passed')

if __name__=='__main__':main()
