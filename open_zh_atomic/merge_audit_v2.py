#!/usr/bin/env python3
from __future__ import annotations

import re
import atomize_v2  # patches atomize.first_request_match before merge_audit imports it
import merge_audit as core

core.ESCAPE_RE = re.compile(r"\\(?:[$%&#_{}]|[ ,;!]|quad\b|qquad\b)")

if __name__ == "__main__":
    raise SystemExit(core.main())
