#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass

DELIMITED_MATH_RE = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$)",
    re.S,
)
ENV_TOKEN_RE = re.compile(r"\\(begin|end)\s*\{([^{}]+)\}")
MATH_ENVS = {
    "equation", "equation*", "align", "align*", "aligned", "array", "cases",
    "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix", "smallmatrix",
    "gather", "gather*", "multline", "multline*", "split", "eqnarray", "eqnarray*",
}
TEX_COMMAND_RE = re.compile(r"\\(?:[A-Za-z]+|.)")
PLACEHOLDER_RE = re.compile(r"\[ANS\]")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?")
SINGLE_VAR_RE = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")
ALLCAPS_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])")
KNOWN_MATH_WORDS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "det", "tr", "rank",
    "gcd", "lcm", "mod", "lim", "sup", "inf", "max", "min", "arg", "ker", "im", "span",
    "diag", "sgn", "ans", "true", "false",
}
WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str


def environment_spans(text: str) -> list[Span]:
    """Return outermost balanced mathematical LaTeX environment spans."""
    stack: list[tuple[str, int]] = []
    spans: list[Span] = []
    for m in ENV_TOKEN_RE.finditer(text):
        typ, env = m.group(1), m.group(2).strip()
        if typ == "begin":
            if env in MATH_ENVS:
                stack.append((env, m.start()))
        elif env in MATH_ENVS and stack:
            # Normal LaTeX nesting closes the top environment. If malformed input has
            # a mismatch, find the nearest matching opener without guessing across it.
            match_index = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == env:
                    match_index = i
                    break
            if match_index is None:
                continue
            start = stack[match_index][1]
            is_outer = match_index == 0
            del stack[match_index:]
            if is_outer:
                spans.append(Span(start, m.end(), "environment"))
    return spans


def primary_math_spans(text: str) -> list[Span]:
    spans = [Span(m.start(), m.end(), "delimited") for m in DELIMITED_MATH_RE.finditer(text)]
    spans.extend(environment_spans(text))
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    merged: list[Span] = []
    for s in spans:
        if not merged or s.start >= merged[-1].end:
            merged.append(s)
        elif s.end > merged[-1].end:
            prev = merged[-1]
            merged[-1] = Span(prev.start, s.end, prev.kind + "+" + s.kind)
    return merged


def mask_spans(text: str, spans: list[Span]) -> str:
    chars = list(text)
    for span in spans:
        for i in range(span.start, span.end):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def mask_primary_math(text: str) -> str:
    return mask_spans(text, primary_math_spans(text))


def extract_primary_math(text: str) -> list[str]:
    return [text[s.start:s.end] for s in primary_math_spans(text)]


def is_math_identifier_word(word: str) -> bool:
    if word.casefold() in KNOWN_MATH_WORDS:
        return True
    if len(word) == 1:
        return True
    if word.isupper() and len(word) <= 8:
        return True
    return False


def natural_words_outside_math(text: str) -> list[str]:
    """English prose words, excluding formulas/placeholders/TeX syntax and variables."""
    plain = mask_primary_math(text)
    plain = PLACEHOLDER_RE.sub(" ", plain)
    plain = TEX_COMMAND_RE.sub(" ", plain)
    plain = NUMBER_RE.sub(" ", plain)
    out: list[str] = []
    for m in WORD_RE.finditer(plain):
        w = m.group(0)
        if is_math_identifier_word(w):
            continue
        out.append(w)
    return out


def has_english_prose(text: str) -> bool:
    return bool(natural_words_outside_math(text))


def critical_tex_sequence(text: str) -> list[str]:
    """Independent syntax skeleton used by final audit for bare TeX formulas."""
    masked = mask_primary_math(text)
    pattern = re.compile(
        r"\\(?:[A-Za-z]+|.)|\[ANS\]|(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?|"
        r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])|(?<![A-Za-z])[A-Z]{2,8}(?![A-Za-z])|"
        r"[=+*/^_<>|{}\[\]()]"
    )
    return pattern.findall(masked)


if __name__ == "__main__":
    sample = r"Consider \begin{equation} I(x)=\int_0^1 x^2\,dx \end{equation} and find $I(2)$."
    spans = extract_primary_math(sample)
    assert len(spans) == 2 and spans[0].startswith(r"\begin{equation}") and spans[1] == "$I(2)$"
    pure = "$3+4$=[ANS]\n$9-3$=[ANS]"
    assert not has_english_prose(pure)
    prose = r"If you rationalize \frac{A}{B}, find A."
    assert has_english_prose(prose)
    print("TeX protection v3 self-test passed")
