#!/usr/bin/env python3
from __future__ import annotations

import re
import translate_shard as core

# TeX spacing/control literals occasionally appear outside $...$, e.g. ``a.e.\\ $x$``.
# They are formatting syntax, not natural language, so keep them out of MT as well.
EXT_ESCAPE_TOKEN = r"\\(?:[$%&#_{}]|[ ,;!]|quad\b|qquad\b)"
core.ESCAPE_TOKEN = EXT_ESCAPE_TOKEN
core.ESCAPE_RE = re.compile(EXT_ESCAPE_TOKEN)
core.PROTECTED_RE = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$|"
    + EXT_ESCAPE_TOKEN + r"|" + core.PLACEHOLDER_TOKEN + r"|" + core.NUMBER_TOKEN + r")",
    re.S,
)

if __name__ == "__main__":
    raise SystemExit(core.main())
