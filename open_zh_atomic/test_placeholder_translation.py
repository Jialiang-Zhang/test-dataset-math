#!/usr/bin/env python3
from __future__ import annotations

import re
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL='Helsinki-NLP/opus-mt-en-zh'
FORMULA='【数学公式占位】'
NUMBER='【数值占位】'
TERM='【数学术语占位】'
SAMPLES=[
    f'Given {FORMULA}, {FORMULA} is its left focus. The line {FORMULA} intersects the two branches. Find the eccentricity.',
    f'A mobile plan charges {NUMBER} dollars for the first {NUMBER} minutes plus {NUMBER} dollars for each additional minute. Write the cost {FORMULA} as a function of {FORMULA}.',
    f'Suppose {FORMULA} are functions such that {FORMULA}. If {FORMULA} converges weakly to {FORMULA}, prove that {FORMULA}.',
    f'Let {TERM} be an {TERM} {TERM} and determine its image.',
    f'For {FORMULA}, compute {FORMULA}; then use {FORMULA} to prove {FORMULA}.'
]
MARKERS=(FORMULA,NUMBER,TERM)


def counts(text:str):
    return {m:text.count(m) for m in MARKERS}


def main():
    torch.set_num_threads(2)
    tok=MarianTokenizer.from_pretrained(MODEL)
    model=MarianMTModel.from_pretrained(MODEL); model.eval()
    enc=tok(SAMPLES,return_tensors='pt',padding=True,truncation=True,max_length=512)
    with torch.inference_mode():
        gen=model.generate(**enc,num_beams=4,do_sample=False,max_new_tokens=512,renormalize_logits=True)
    outs=tok.batch_decode(gen,skip_special_tokens=True)
    for src,dst in zip(SAMPLES,outs):
        expected=counts(src); got=counts(dst)
        print('SRC:',src); print('DST:',dst); print('EXPECTED',expected); print('GOT',got); print()
        assert expected==got,(src,dst,expected,got)
    # At least the key geometry wording should remain semantically sensible in context.
    assert '焦点' in outs[0], outs[0]
    assert ('偏心' in outs[0] or '离心' in outs[0]), outs[0]
    print('Chinese generic placeholder preservation probe passed')

if __name__=='__main__':main()
