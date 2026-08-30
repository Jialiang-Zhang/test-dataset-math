#!/usr/bin/env python3
from __future__ import annotations

import atomize_v2  # installs guarded interrogative detection
import atomize as core
from tex_protection_v3 import mask_primary_math, extract_primary_math

# Every splitting heuristic consults core.mask_math at call time. Replacing it here
# makes explicit labels/options inside equation/align/array/cases/matrix environments
# invisible to the question splitter while leaving surrounding prose available.
core.mask_math = mask_primary_math


def self_test() -> None:
    atomize_v2.self_test()
    options = r"""Which value is correct?
\begin{align*}
\text{A)}\ & 1 &
\text{B)}\ & 2\\
\text{C)}\ & 3 &
\text{D)}\ & 4
\end{align*}"""
    parts, reason = core.split_atomic_problem(options)
    assert len(parts) == 1, (reason, parts)
    assert len(extract_primary_math(options)) == 1


if __name__ == "__main__":
    self_test()
    raise SystemExit(core.main())
