#!/usr/bin/env python3
"""Convert Unlimited-OCR 2 raw output into MarathonDocBench evaluation Markdown.

Unlimited-OCR 2 does NOT emit Markdown. Each document is a single string of
layout blocks:

    <PAGE><BLOCK>title<LEVEL1>[x1,y1,x2,y2]<FORK>text ...
    <BLOCK>text[x1,y1,x2,y2]<FORK>body ...
    <BLOCK>cross_table[bbox]<FORK><table>...</table><PAGE><BLOCK>append_table[bbox]<FORK>...

Conversion rules (format -> Markdown):

  * split on `<PAGE>`; keep blank pages so page numbers don't shift
  * `title<LEVELn>`        -> `#`*n heading (missing level -> one `#`)
  * `image`/`chart`/...    -> `![](pageN_x1_y1_x2_y2.jpg)` (coords clamped 0-1000)
  * `table`/`append_table` -> HTML; cross-page fragments merged (repeated header de-dup)
  * `cross_<t>`/`append_<t>`/`<MERGE>` -> grouped (union-find) and concatenated in
    reading order, matching the GT truncated-merge semantics
  * strip stray tokens; keep the heading tree single-rooted (matches the GT)
  * strip the `<!-- looping output truncated by loop guard -->` marker left by
    `infer.py --loop-guard` (an annotation, not document content; literal kept
    in sync with `repetition_guard.TRUNCATION_MARKER` by the tests)

Usage
-----
    python infer_utils/process_unlimitedOCR2.py \
        --infer-dir <dir of raw <doc_id>.txt> \
        --out-dir   <dir of <doc_id>.md>
"""
import argparse
import os
import re

# ── block / token patterns ───────────────────────────────────────────────────
PAGE_SPLIT_RE = re.compile(r'<\s*/?\s*PAGE\s*\d*\s*>', re.I)
MERGE_RE = re.compile(r'<\s*/?\s*MERGE\s*>', re.I)
STRAY_TOKEN_RE = re.compile(
    r'<[|｜]/?[a-zA-Z_]+[|｜]>|<image>'
    r'|<\s*/?\s*PAGE\s*\d*\s*>|<\s*/?\s*BLOCK\s*>|<\s*/?\s*FORK\s*>'
    r'|<\s*/?\s*LEVEL\s*\d*\s*[>}]?|<\s*/?\s*MERGE\s*>', re.I)
# <BLOCK>label [<LEVELn>|[bbox]] [<LEVELn>|[bbox]] [<FORK>] ; LEVEL-first or bbox-first.
BLOCK_RE = re.compile(
    r'<\s*BLOCK\s*>\s*([a-zA-Z_]+)'
    r'(?:'
    r'\s*<\s*/?\s*LEVEL\s*(\d*)\s*[>}]?(?:\s*\[\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)'
    r'(?:\s*[,，]\s*-?\d+)*\s*\])?'
    r'|'
    r'\s*\[\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)'
    r'(?:\s*[,，]\s*-?\d+)*\s*\](?:\s*<\s*/?\s*LEVEL\s*(\d*)\s*[>}]?)?'
    r')?'
    r'\s*(?:<\s*FORK\s*>)?', re.I)
TR_RE = re.compile(r'<tr\b[^>]*>.*?</tr>', re.I | re.S)
TABLE_OPEN_RE = re.compile(r'<table\b[^>]*>', re.I)
SPAN_ATTR_RE = re.compile(r'\b(colspan|rowspan)\s*=\s*"([^"]*)"', re.I)
# annotation line appended by infer.py --loop-guard; never document content
LOOP_GUARD_MARKER_RE = re.compile(
    r'[ \t]*<!-- looping output truncated by loop guard -->[ \t]*\n?')

DROP_LABELS = set()                       # header/footer/page_number kept (canonical default)
IMAGE_LABELS = {'image', 'figure', 'picture', 'chart', 'img', 'seal'}
NOISE_CONTENT = {'[non-text]', '[unreadable]', '[image]', '[figure]'}
CHAIN_TYPES = {'text', 'table'}
MAX_HEADER_ROWS = 5

_BASE_LABELS = {
    'text', 'title', 'header', 'footer', 'page_number', 'image', 'figure',
    'chart', 'seal', 'table', 'image_caption', 'image_footnote',
    'page_footnote', 'equation', 'ref_text', 'aside_text', 'algorithm',
}
_BARE_LABELS = _BASE_LABELS | {
    f'{p}{l}' for p in ('cross_', 'append_') for l in _BASE_LABELS}
BARE_BLOCK_RE = re.compile(
    r'(?:(?<=<FORK>)|(?<=<PAGE>)|(?<=\n)|^)'
    r'(?:<[|｜]\s*det\s*[|｜]>)?'
    r'((?:[a-zA-Z_][a-zA-Z_0-9]*)(?:\s*<\s*/?\s*LEVEL\s*\d*\s*[>}]?)?\s*'
    r'\[\s*-?\d+(?:\s*[,，]\s*-?\d+){3,}[^\]]*\])', re.I)


def _scale(v):
    return max(0, min(1000, int(v)))


def _fix_span_attrs(text):
    def repl(m):
        name, val = m.group(1), m.group(2)
        try:
            int(val)
            return m.group(0)
        except ValueError:
            num = re.search(r'\d+', val)
            return f'{name}="{num.group() if num else 1}"'
    return SPAN_ATTR_RE.sub(repl, text)


def _fix_bare_blocks(text):
    """Add the missing <BLOCK> prefix to bare block declarations (known labels only)."""
    def _repl(m):
        label = re.match(r'[a-zA-Z_][a-zA-Z_0-9]*', m.group(1)).group(0)
        if label.lower() not in _BARE_LABELS:
            return m.group(0)
        return f'<BLOCK>{m.group(1)}'
    text = BARE_BLOCK_RE.sub(_repl, text)
    return re.sub(r'\n\s*<FORK>\s*(?=<BLOCK>)', '\n', text, flags=re.I)


def _base_type(label):
    """cross_text/append_text -> text; append_table/cross_table -> table."""
    for p in ('append_', 'cross_'):
        if label.startswith(p):
            return label[len(p):]
    return label


def _parse_blocks(text, page_base=1):
    """Split on <PAGE>; return blocks [{label,bbox,level,content,page,merge_prev}]."""
    text = _fix_bare_blocks(text)
    segments = PAGE_SPLIT_RE.split(text)
    if segments and not segments[0].strip():
        segments = segments[1:]
    blocks = []
    for i, seg in enumerate(segments):
        page_no = page_base + i
        if not seg.strip():
            continue
        matches = list(BLOCK_RE.finditer(seg))
        if not matches:
            blocks.append({'label': '__raw__', 'bbox': None, 'level': None,
                           'content': MERGE_RE.sub('', seg).strip(),
                           'page': page_no, 'merge_prev': False})
            continue
        for j, m in enumerate(matches):
            gap = seg[:m.start()] if j == 0 else seg[matches[j - 1].end():m.start()]
            merge_prev = bool(MERGE_RE.search(gap))
            if j == 0:
                pre = STRAY_TOKEN_RE.sub('', MERGE_RE.sub('', gap)).strip()
                if pre:
                    blocks.append({'label': '__raw__', 'bbox': None, 'level': None,
                                   'content': pre, 'page': page_no, 'merge_prev': False})
            end = matches[j + 1].start() if j + 1 < len(matches) else len(seg)
            content = MERGE_RE.sub('', seg[m.end():end]).strip()
            if m.group(2) is not None:          # LEVEL-first
                level = int(m.group(2)) if m.group(2) else None
                bbox = (tuple(int(m.group(k)) for k in range(3, 7))
                        if m.group(3) is not None else None)
            else:                                # bbox-first (or none)
                bbox = (tuple(int(m.group(k)) for k in range(7, 11))
                        if m.group(7) is not None else None)
                level = int(m.group(11)) if m.group(11) else None
            blocks.append({'label': m.group(1).lower(), 'bbox': bbox, 'level': level,
                           'content': content, 'page': page_no, 'merge_prev': merge_prev})
    return blocks

def _merge_adjacent_titles(blocks):
    """Merge strictly-adjacent, same-page, same-<LEVELn> title blocks into one."""
    if not blocks:
        return blocks
    out = [blocks[0]]
    for blk in blocks[1:]:
        prev = out[-1]
        if (blk['label'] == 'title' and prev['label'] == 'title'
                and blk['page'] == prev['page']
                and blk['level'] is not None and prev['level'] is not None
                and blk['level'] == prev['level']):
            prev['content'] = '\n'.join(x for x in (prev['content'], blk['content']) if x)
            continue
        out.append(blk)
    return out


def _build_groups(blocks):
    """Union-find grouping of append/cross chains and same-page <MERGE> columns."""
    n = len(blocks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pages_with = {}
    for blk in blocks:
        b = _base_type(blk['label'])
        if blk['label'] != '__raw__' and b in CHAIN_TYPES:
            pages_with.setdefault(b, set()).add(blk['page'])

    def _table_adjacent(tail_page, cur_page):
        if cur_page - tail_page <= 1:
            return True
        return all(p not in pages_with.get('table', ())
                   for p in range(tail_page + 1, cur_page))

    open_chain = {}                       # base type -> current chain-tail index
    for i, blk in enumerate(blocks):
        label, base = blk['label'], _base_type(blk['label'])
        if label == '__raw__' or base not in CHAIN_TYPES:
            continue
        if label.startswith('append_'):
            tail = open_chain.get(base)
            if tail is None:
                continue
            if base == 'table' and not _table_adjacent(blocks[tail]['page'], blk['page']):
                open_chain.pop(base, None)
                continue
            union(tail, i)
            open_chain[base] = i
        elif label.startswith('cross_'):
            open_chain[base] = i
        elif base == 'table':
            open_chain[base] = i

    for i, blk in enumerate(blocks):       # same-page <MERGE> column edges
        if not blk.get('merge_prev') or i == 0:
            continue
        pblk = blocks[i - 1]
        if pblk['page'] != blk['page']:
            continue
        if (blk['label'] == 'title' or _base_type(blk['label']) == 'text') \
                and (pblk['label'] == 'title' or _base_type(pblk['label']) == 'text'):
            union(i - 1, i)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[r] for r in sorted(groups, key=lambda r: groups[r][0])]


def _row_key(tr_html):
    txt = re.sub(r'<[^>]+>', ' ', tr_html)
    txt = ''.join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in txt)
    ncells = len(re.findall(r'<t[hd]\b', tr_html, re.I))
    return (ncells, ''.join(txt.split()).lower())


def _merge_table_fragments(frags):
    """Merge cross-page table fragments, skipping repeated leading header rows."""
    frags = [f for f in frags if f and f.strip()]
    if len(frags) <= 1:
        return frags[0] if frags else ''
    base_rows = TR_RE.findall(frags[0])
    if not base_rows:
        return '\n\n'.join(frags)
    out_rows, base_keys, extra = list(base_rows), [_row_key(r) for r in base_rows], []
    for frag in frags[1:]:
        rows = TR_RE.findall(frag)
        if not rows:
            extra.append(frag.strip())
            continue
        skip = 0
        for r in rows[:MAX_HEADER_ROWS]:
            if skip < len(base_keys) and _row_key(r) == base_keys[skip]:
                skip += 1
            else:
                break
        out_rows.extend(rows[skip:])
    m = TABLE_OPEN_RE.search(frags[0])
    merged = (m.group(0) if m else '<table>') + ''.join(out_rows) + '</table>'
    return merged + ('\n\n' + '\n\n'.join(extra) if extra else '')

def _render_blocks(blocks, groups):
    """Render each group to a Markdown item (figure/table/heading/text)."""
    items = []
    for members in groups:
        head = blocks[members[0]]
        base = head['label'] if head['label'] == '__raw__' else _base_type(head['label'])
        contents = [blocks[k]['content'] for k in members
                    if blocks[k]['content'] and blocks[k]['content'].lower() not in NOISE_CONTENT]
        if base == '__raw__':
            if contents:
                items.append(contents[0])
            continue
        if base in IMAGE_LABELS:
            if head['bbox'] is not None:
                x1, y1, x2, y2 = (_scale(v) for v in head['bbox'])
                items.append(f'![](page{head["page"]}_{x1}_{y1}_{x2}_{y2}.jpg)')
            if contents:
                items.append(contents[0])
            continue
        if not contents or base in DROP_LABELS:
            continue
        if base == 'title':
            lv = max(1, head['level'] or 1)
            joined = ''.join(contents)
            if not STRAY_TOKEN_RE.sub('', joined).strip():
                continue
            items.append(f'{"#" * lv} {joined}')
            continue
        if base == 'table':
            items.append(_merge_table_fragments(contents) if len(contents) > 1 else contents[0])
            continue
        items.append(''.join(contents))       # text family: concatenate, no separator
    return items


def _head_level(item):
    s = item.lstrip()
    return len(s) - len(s.lstrip('#'))


def _single_root_titles(items):
    """Collapse multiple top-level headings under the first one (single root)."""
    idxs = [i for i, it in enumerate(items) if it.lstrip().startswith('#')]
    if len(idxs) <= 1:
        return items
    levels = [_head_level(items[i]) for i in idxs]
    min_lv = min(levels)
    if sum(1 for lv in levels if lv == min_lv) <= 1:
        return items
    first_lv, out = levels[0], list(items)
    for k, i in enumerate(idxs):
        body = items[i].lstrip()[levels[k]:].lstrip(' ')
        new_lv = 1 if k == 0 else max(2, levels[k] - first_lv + 2)
        out[i] = '#' * new_lv + ' ' + body
    return out


def _fold_head_ws(items):
    out = []
    for it in items:
        s = it.lstrip()
        if s.startswith('#'):
            n = len(s) - len(s.lstrip('#'))
            it = '#' * n + ' ' + re.sub(r'\s+', ' ', s[n:]).strip()
        out.append(it)
    return out


def _drop_empty_heads(items):
    return [it for it in items
            if not (it.lstrip().startswith('#') and not it.lstrip().lstrip('#').strip())]


def convert(result):
    """Unlimited-OCR 2 raw document string -> MarathonDocBench evaluation Markdown."""
    text = (result or '').strip()
    text = LOOP_GUARD_MARKER_RE.sub('', text)  # loop-guard annotation
    if text.startswith('```markdown'):
        text = text[len('```markdown'):].strip()
    if text.endswith('```'):
        text = text[:-3].strip()

    blocks = _parse_blocks(text, page_base=1)
    blocks = _merge_adjacent_titles(blocks)
    groups = _build_groups(blocks)
    items = _drop_empty_heads(_render_blocks(blocks, groups))
    items = _fold_head_ws(items)
    text = '\n\n'.join(items)

    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    text = STRAY_TOKEN_RE.sub('', text)
    text = _fix_span_attrs(text)
    text = re.sub(r'(?m)^\s*\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]\s*$', '', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return '\n\n'.join(_single_root_titles(text.split('\n\n')))

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--infer-dir', required=True,
                    help='directory of raw per-document outputs (<doc_id>.txt)')
    ap.add_argument('--out-dir', required=True,
                    help='output directory of <doc_id>.md')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(args.infer_dir)):
        if not name.endswith(('.txt', '.md')):
            continue
        with open(os.path.join(args.infer_dir, name), encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        doc_id = os.path.splitext(name)[0]
        with open(os.path.join(args.out_dir, f'{doc_id}.md'), 'w', encoding='utf-8') as f:
            f.write(convert(raw))
        n += 1
    print(f'[process_unlimitedOCR2] converted {n} documents -> {args.out_dir}')


if __name__ == '__main__':
    main()
