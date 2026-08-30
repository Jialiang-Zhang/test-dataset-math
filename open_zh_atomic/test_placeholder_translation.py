#!/usr/bin/env python3
from __future__ import annotations

import re
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL='Helsinki-NLP/opus-mt-en-zh'
SAMPLES=[
    'Given ZXQMAAAAQXZ, ZXQMAAABQXZ is its left focus. The line ZXQMAAACQXZ intersects the two branches. Find the eccentricity.',
    'A mobile plan charges ZXQNAAAAQXZ dollars for the first ZXQNAAABQXZ minutes plus ZXQNAAACQXZ dollars for each additional minute. Write the cost ZXQVAAAAQXZ as a function of ZXQVAAABQXZ.',
    'Suppose ZXQMAAAAQXZ are functions such that ZXQMAAABQXZ. If ZXQMAAACQXZ converges weakly to ZXQMAAADQXZ, prove that ZXQMAAAEQXZ.',
    'Let ZXQTAAAAQXZ be an ZXQTAAABQXZ ZXQTAAACQXZ and determine its image.'
]
MARKER_RE=re.compile(r'ZXQ[MNVT][A-Z]{4}QXZ')


def main():
    torch.set_num_threads(2)
    tok=MarianTokenizer.from_pretrained(MODEL)
    model=MarianMTModel.from_pretrained(MODEL); model.eval()
    enc=tok(SAMPLES,return_tensors='pt',padding=True,truncation=True,max_length=512)
    with torch.inference_mode():
        gen=model.generate(**enc,num_beams=4,do_sample=False,max_new_tokens=512,renormalize_logits=True)
    outs=tok.batch_decode(gen,skip_special_tokens=True)
    for src,dst in zip(SAMPLES,outs):
        expected=MARKER_RE.findall(src); got=MARKER_RE.findall(dst)
        print('SRC:',src); print('DST:',dst); print('EXPECTED',expected); print('GOT',got); print()
        assert sorted(expected)==sorted(got),(src,dst,expected,got)
    print('pure-letter placeholder preservation probe passed')

if __name__=='__main__':main()
