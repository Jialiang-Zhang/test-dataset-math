#!/usr/bin/env python3
from __future__ import annotations

import re
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL='Helsinki-NLP/opus-mt-en-zh'
SAMPLES=[
    'Given ZXQMATH0000QXZ, ZXQMATH0001QXZ is its left focus. The line ZXQMATH0002QXZ intersects the two branches. Find the eccentricity.',
    'A mobile plan charges ZXQNUM0000QXZ dollars for the first ZXQNUM0001QXZ minutes plus ZXQNUM0002QXZ dollars for each additional minute. Write the cost ZXQVAR0000QXZ as a function of ZXQVAR0001QXZ.',
    'Suppose ZXQMATH0000QXZ are functions such that ZXQMATH0001QXZ. If ZXQMATH0002QXZ converges weakly to ZXQMATH0003QXZ, prove that ZXQMATH0004QXZ.',
    'Let ZXQTERM0000QXZ be an ZXQTERM0001QXZ ZXQTERM0002QXZ and determine its image.'
]


def main():
    torch.set_num_threads(2)
    tok=MarianTokenizer.from_pretrained(MODEL)
    model=MarianMTModel.from_pretrained(MODEL); model.eval()
    enc=tok(SAMPLES,return_tensors='pt',padding=True,truncation=True,max_length=512)
    with torch.inference_mode():
        gen=model.generate(**enc,num_beams=4,do_sample=False,max_new_tokens=512)
    outs=tok.batch_decode(gen,skip_special_tokens=True)
    marker=re.compile(r'ZXQ(?:MATH|NUM|VAR|TERM)\d{4}QXZ')
    for src,dst in zip(SAMPLES,outs):
        expected=marker.findall(src); got=marker.findall(dst)
        print('SRC:',src); print('DST:',dst); print('EXPECTED',expected); print('GOT',got); print()
        assert sorted(expected)==sorted(got),(src,dst,expected,got)
    print('placeholder preservation probe passed')

if __name__=='__main__':main()
