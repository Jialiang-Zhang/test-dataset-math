#!/usr/bin/env python3
from __future__ import annotations

import re
import atomize_v4  # installs v4 rules into core
import atomize as core

BASE_SPLIT = core.split_atomic_problem
ANS = r"\[ANS\]"
NUMBERED_ANS_RE = re.compile(rf"{ANS}\s*(?P<num>\d{{1,3}})\s*[\.)]\s*")
INLINE_ALPHA_RE = re.compile(r"(?i)(?<![A-Za-z])(?P<label>[a-h])\)\s+")
INLINE_ROMAN_RE = re.compile(r"(?i)(?<![A-Za-z])(?P<label>i{1,3}|iv|v|vi{0,3}|ix|x)\)\s+")
EACH_LIST_RE = re.compile(
    r"(?i)(?:each\s+of\s+the\s+following|for\s+each\s+of\s+the\s+following|for\s+each|"
    r"evaluate\s+the\s+following|solve\s+each|match\s+(?:each|the)|indicate\s+whether\s+the\s+following|"
    r"select\s+true\s+or\s+false|enter\s+(?:a\s+)?t\s+or\s+(?:an\s+)?f|"
    r"express\s+each\s+of\s+the\s+following|complete\s+the\s+table|fill\s+in\s+the\s+blanks\s+below)|"
    r"(?:下列各|以下各|逐项|每一项|分别|判断下列|完成下表|填写下列各空)"
)
STRUCTURED_SINGLE_RE = re.compile(
    r"(?i)(?:ordered\s+pair|one\s+pair|coordinates?\s+of\s+the\s+point|rectangular\s+coordinates|"
    r"find\s+(?:the\s+)?matrix|find\s+(?:the\s+)?vector|components?\s+of|"
    r"in\s+the\s+form\s+\$?u\+wt|coefficient\s+\$?c\$?\s+.*exponent\s+\$?e\$?|"
    r"piecewise[- ]defined\s+(?:linear\s+)?function|rewrite\s+the\s+following\s+using\s+a\s+single\s+exponent)|"
    r"(?:有序对|点的坐标|直角坐标|求矩阵|求向量|分量|写成.*u\+wt|分段函数)"
)
OPTION_START_RE = re.compile(r"(?<![A-Za-z])A\.\s+")
LABEL_ITEM_RE = re.compile(r"(?<![A-Za-z])(?P<label>[A-Z][A-Za-z][A-Za-z /-]{0,35}):\s*")


def _sequential_ints(matches) -> bool:
    vals=[int(m.group('num')) for m in matches]
    return len(vals)>=2 and all(vals[i+1]==vals[i]+1 for i in range(len(vals)-1))


def _find_options_start(text: str, after: int) -> int | None:
    # Options are a shared suffix only when A. and B. both occur after the final task.
    for m in OPTION_START_RE.finditer(text, after):
        tail=text[m.start():]
        if re.search(r"(?<![A-Za-z])B\.\s+", tail):
            return m.start()
    return None


def split_numbered_answer_items(text: str):
    matches=list(NUMBERED_ANS_RE.finditer(text))
    if len(matches)<2 or not _sequential_ints(matches):
        return None
    prefix=text[:matches[0].start()].strip()
    options_start=_find_options_start(text, matches[-1].end())
    common_suffix=text[options_start:].strip() if options_start is not None else ""
    final_end=options_start if options_start is not None else len(text)
    out=[]
    for i,m in enumerate(matches):
        end=matches[i+1].start() if i+1<len(matches) else final_end
        body=text[m.end():end].strip()
        if not body:
            continue
        child=f"{prefix}\n{body}".strip() if prefix else body
        if common_suffix:
            child=f"{child}\n{common_suffix}".strip()
        out.append((f"ansitem{m.group('num')}",child))
    return out if len(out)>=2 else None


def _masked_marker_matches(text: str, pattern: re.Pattern[str]):
    masked=core.mask_math(text)
    return list(pattern.finditer(masked))


def split_inline_lettered_items(text: str):
    if not EACH_LIST_RE.search(core.mask_math(text)):
        return None
    for pattern,kind in ((INLINE_ALPHA_RE,'alpha'),(INLINE_ROMAN_RE,'roman')):
        matches=_masked_marker_matches(text,pattern)
        if len(matches)<2:
            continue
        # Require sequential labels to avoid prose like "a)" references.
        if kind=='alpha':
            vals=[ord(m.group('label').lower())-96 for m in matches]
        else:
            table={'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6,'vii':7,'viii':8,'ix':9,'x':10}
            vals=[table.get(m.group('label').lower(),-99) for m in matches]
        if not all(vals[i+1]==vals[i]+1 for i in range(len(vals)-1)):
            continue
        prefix=text[:matches[0].start()].strip()
        out=[]
        for i,m in enumerate(matches):
            end=matches[i+1].start() if i+1<len(matches) else len(text)
            body=text[m.end():end].strip()
            if body:
                child=f"{prefix}\n{body}".strip() if prefix else body
                out.append((f"inline{m.group('label')}",child))
        if len(out)>=2:
            return out
    return None


def split_answer_sentences(text: str):
    if text.count('[ANS]')<2 or STRUCTURED_SINGLE_RE.search(core.mask_math(text)):
        return None
    # High-confidence list/evaluation prompts: split clauses ending in one answer slot.
    if not EACH_LIST_RE.search(core.mask_math(text)):
        return None
    spans=[];start=0
    for m in re.finditer(r"\[ANS\]\s*[\.,;]?",text):
        end=m.end();piece=text[start:end].strip()
        if '[ANS]' in piece:
            spans.append((start,end,piece))
        start=end
    if len(spans)<2:
        return None
    # Shared prefix is everything before the first task-like mathematical/content clause.
    first_ans=text.find('[ANS]')
    prefix=text[:first_ans]
    # Walk backward to the most recent sentence/colon separator; preceding prose is shared.
    cuts=[prefix.rfind('. '),prefix.rfind(': '),prefix.rfind('\n')]
    cut=max(cuts)
    shared=prefix[:cut+1].strip() if cut>=0 else ""
    first_body_start=cut+1 if cut>=0 else 0
    out=[]
    prev=first_body_start
    for i,m in enumerate(re.finditer(r"\[ANS\]\s*[\.,;]?",text),1):
        piece=text[prev:m.end()].strip()
        if '[ANS]' not in piece:
            prev=m.end();continue
        child=f"{shared}\n{piece}".strip() if shared else piece
        out.append((f"anssentence{i}",child))
        prev=m.end()
    return out if len(out)>=2 else None


def split_labelled_definition_items(text: str):
    if text.count('[ANS]')<2 or not re.search(r"(?i)fill\s+in\s+the\s+blanks",text):
        return None
    matches=list(LABEL_ITEM_RE.finditer(text))
    candidates=[]
    for i,m in enumerate(matches):
        end=matches[i+1].start() if i+1<len(matches) else len(text)
        body=text[m.start():end].strip()
        if body.count('[ANS]')==1:
            candidates.append((m,end,body))
    if len(candidates)<2:
        return None
    prefix=text[:candidates[0][0].start()].strip()
    out=[]
    for i,(_,_,body) in enumerate(candidates,1):
        child=f"{prefix}\n{body}".strip() if prefix else body
        out.append((f"definition{i}",child))
    return out if len(out)>=2 else None


def split_repeated_for_items(text: str):
    # Example: "For line1, slope=[ANS] and intercept=[ANS]. For line2, ..."
    if text.count('[ANS]')<4 or not re.search(r"(?i)for\s+each\s+of\s+the\s+following",text):
        return None
    matches=list(re.finditer(r"(?i)(?<![A-Za-z])For\s+",text))
    # First "for each" is directive; item starts are later For ... clauses.
    matches=[m for m in matches if not re.match(r"(?i)For\s+each\b",text[m.start():m.start()+30])]
    if len(matches)<2:
        return None
    prefix=text[:matches[0].start()].strip()
    out=[]
    for i,m in enumerate(matches):
        end=matches[i+1].start() if i+1<len(matches) else len(text)
        body=text[m.start():end].strip()
        if '[ANS]' in body:
            child=f"{prefix}\n{body}".strip() if prefix else body
            out.append((f"foritem{i+1}",child))
    return out if len(out)>=2 else None


def split_atomic_problem_v5(text: str):
    parts,reason=BASE_SPLIT(text)
    if len(parts)>1:
        return parts,reason
    for reason2,fn in (
        ('numbered_answer_items',split_numbered_answer_items),
        ('inline_lettered_items',split_inline_lettered_items),
        ('labelled_definition_items',split_labelled_definition_items),
        ('repeated_for_items',split_repeated_for_items),
        ('answer_sentences',split_answer_sentences),
    ):
        extra=fn(text)
        if extra:
            return extra,reason2
    return parts,reason

core.split_atomic_problem=split_atomic_problem_v5


def self_test():
    numbered='Match the statements. [ANS] 1. first [ANS] 2. second A. alpha B. beta'
    p,r=core.split_atomic_problem(numbered);assert len(p)==2 and r=='numbered_answer_items',(r,p)
    inline=r'Solve each equation: a) $x+1=2$, $x=$ [ANS]. b) $y+2=4$, $y=$ [ANS].'
    p,r=core.split_atomic_problem(inline);assert len(p)==2 and r=='inline_lettered_items',(r,p)
    defs='Fill in the blanks below Term: a number or a [ANS] Coefficient: the number multiplying a [ANS]'
    p,r=core.split_atomic_problem(defs);assert len(p)==2 and r=='labelled_definition_items',(r,p)
    each=r'Evaluate the following arithmetic expressions. $4^2-8=$ [ANS]. $6^2-12=$ [ANS].'
    p,r=core.split_atomic_problem(each);assert len(p)==2 and r=='answer_sentences',(r,p)
    structured=r'Find one pair $(x,y)$. $x=$ [ANS] $y=$ [ANS]'
    p,r=core.split_atomic_problem(structured);assert len(p)==1,(r,p)

if __name__=='__main__':
    self_test()
    raise SystemExit(core.main())
