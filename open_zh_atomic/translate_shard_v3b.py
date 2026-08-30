#!/usr/bin/env python3
from __future__ import annotations

import translate_shard_v3 as core
from tex_protection_v3 import mask_primary_math

_original_audit = core.audit_glossary
core.audit_glossary = lambda source, translated: _original_audit(mask_primary_math(source), translated)

if __name__ == "__main__":
    raise SystemExit(core.main())
