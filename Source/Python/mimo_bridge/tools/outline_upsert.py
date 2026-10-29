# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Insert (or replace) a named HTML section in the outline, right before the Sources footer."""
import sys
OUTLINE = "docs/MIMo_Testbed_Implementation_Outline_2026-08-19.html"


def upsert(section_id: str, html: str, outline: str = OUTLINE):
    s = open(outline, encoding="utf-8").read()
    start = f'<!-- BEGIN {section_id} -->'
    end = f'<!-- END {section_id} -->'
    block = f'{start}\n{html}\n{end}\n'
    if start in s:
        s = s[:s.index(start)] + block + s[s.index(end) + len(end) + 1:]
    else:
        marker = '<div class="foot">'
        s = s.replace(marker, block + marker, 1)
    open(outline, "w", encoding="utf-8", newline="\n").write(s)


if __name__ == "__main__":
    sid = sys.argv[1]
    html = open(sys.argv[2], encoding="utf-8").read()
    upsert(sid, html)
    print("upserted", sid)
