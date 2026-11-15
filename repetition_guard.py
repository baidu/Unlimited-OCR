"""Detect and bound pathological repetition in OCR model output.

Some pages send OCR models into a degeneration loop: the same structural unit
is re-emitted until the generation budget is exhausted (issue #55).  The loops
observed on real pages are *variable-tail* repetitions -- each iteration
differs in a few tokens, typically bounding-box coordinates inside
``<|det|>...<|/det|>`` blocks or an incrementing counter -- so no exact-match
n-gram processor of any size can ban them, and the failure only becomes
visible in the decoded output.

This module works on decoded text only.  It is independent of the inference
backend (transformers, SGLang, ...), of the model version, of the language and
of the hardware.  It provides:

* :func:`is_looping_output` -- pure detection;
* :func:`looping_onset` -- character offset where the repetition run starts;
* :func:`truncate_looping_output` -- bounded output: cut at the repetition
  onset and append an explicit marker instead of unbounded garbage;
* a small command line interface to check/truncate raw output files.

The two detection signals and their default thresholds (>= 30 consecutive
repeats, more than 2,000 characters) were calibrated on OmniDocBench
v1.6 pages:
reproduced loops measured 46 and 1,325 consecutive repeats while normal pages
never exceeded 1 and 4.  Validation provenance: AMD gfx1151, Windows ROCm,
transformers 4.52.1, greedy decoding (issue #55 evidence chain).  Detection is
a heuristic: content with >= 30 *legitimately* consecutive identical units
(rare; not observed in the calibration corpus) would also be flagged; keep the
guard disabled for such inputs.  Repetition without any ``<|det|>`` structure
*and* without separators is a known blind spot.
"""

import argparse
import re
import sys

LOOPING_MIN_CHARS = 2000
LOOPING_BLOCK_REPEATS = 30
LOOPING_PATTERN_REPEATS = 30
TRUNCATION_MARKER = "<!-- looping output truncated by loop guard -->"

_DET_SPLIT_RE = re.compile(r"(<\|det\|>.*?<\|/det\|>)", re.DOTALL)
_DET_BOX_RE = re.compile(r"(<\|det\|>\s*[\w-]+\s*)\[[^\]]*\](\s*<\|/det\|>)")
_SEG_SPLIT_RE = re.compile(r"[，,;；\n\r]+|(?=<\|det\|>)|(?<=<\|/det\|>)")
_DIGIT_RE = re.compile(r"\d+")


def _max_consecutive_repeats(items):
    """Length of the longest run of equal consecutive items."""
    best = cur = 0
    prev = object()
    for item in items:
        cur = cur + 1 if item == prev else 1
        prev = item
        if cur > best:
            best = cur
    return best


def _block_units(text):
    """Split output into units: det block (bbox masked) + following free text.

    Loop iterations differ from each other mainly in bbox coordinate tokens
    (and occasionally in a digit or a case), so coordinates are masked before
    comparison while the recognized content itself stays part of the unit.
    """
    parts = _DET_SPLIT_RE.split(text)
    units = []
    for i in range(1, len(parts), 2):
        tail = parts[i + 1] if i + 1 < len(parts) else ''
        nxt = tail.find('<|det|>')
        content = tail if nxt < 0 else tail[:nxt]
        units.append(
            _DET_BOX_RE.sub(r'\1[BBOX]\2', parts[i]) + content.strip())
    return units


def is_looping_output(text, min_chars=LOOPING_MIN_CHARS,
                      block_repeats=LOOPING_BLOCK_REPEATS,
                      pattern_repeats=LOOPING_PATTERN_REPEATS):
    """Return True if ``text`` shows pathological repetition (issue #55).

    Two signals, either of which is sufficient:

    1. structure-normalized units (det block with the bbox masked plus its
       content) repeating consecutively at least ``block_repeats`` times;
    2. digit-masked segments repeating consecutively at least
       ``pattern_repeats`` times (counter loops such as
       ``3271年出生，3272年出生，...``).

    Failure semantics: never raises; empty or None input and outputs shorter
    than ``min_chars`` return False (no opinion).
    """
    if not text or len(text) <= min_chars:
        return False
    units = _block_units(text)
    if units and _max_consecutive_repeats(units) >= block_repeats:
        return True
    segs = [s.strip() for s in _SEG_SPLIT_RE.split(text) if s.strip()]
    masked = [_DIGIT_RE.sub('#', s) for s in segs]
    return bool(masked) and _max_consecutive_repeats(masked) >= pattern_repeats


def looping_onset(text, min_chars=LOOPING_MIN_CHARS,
                  block_repeats=LOOPING_BLOCK_REPEATS,
                  pattern_repeats=LOOPING_PATTERN_REPEATS):
    """Return the char offset where the qualifying repetition run starts.

    None when ``text`` is empty or contains no qualifying run (same repeat
    thresholds as :func:`is_looping_output`; unlike it, this function does
    not apply the ``min_chars`` length gate -- callers gate on
    :func:`is_looping_output` first).  The block signal is preferred: when
    both signals fire, the earliest block-run start is returned.  Never
    raises.
    """
    if not text:
        return None
    parts = _DET_SPLIT_RE.split(text)
    units = []
    pos = len(parts[0]) if parts else 0
    for i in range(1, len(parts), 2):
        tail = parts[i + 1] if i + 1 < len(parts) else ''
        nxt = tail.find('<|det|>')
        content = tail if nxt < 0 else tail[:nxt]
        key = _DET_BOX_RE.sub(r'\1[BBOX]\2', parts[i]) + content.strip()
        units.append((key, pos))
        pos += len(parts[i]) + (nxt if nxt >= 0 else len(tail))
    best_start, best_run, run_start, run = None, 0, None, 0
    prev = object()
    for key, start in units:
        if key == prev:
            run += 1
            if run >= block_repeats and run > best_run:
                best_run, best_start = run, run_start
        else:
            run_start, run = start, 1
        prev = key
    if best_start is not None:
        return best_start
    segs, seg_starts, pos = [], [], 0
    for seg in _SEG_SPLIT_RE.split(text):
        pos_s = text.find(seg, pos)
        if pos_s < 0:
            continue
        if seg.strip():
            segs.append(_DIGIT_RE.sub('#', seg.strip()))
            seg_starts.append(pos_s)
        pos = pos_s + len(seg)
    best_start, best_run, run_start, run = None, 0, None, 0
    prev = object()
    for key, start in zip(segs, seg_starts, strict=True):
        if key == prev:
            run += 1
            if run >= pattern_repeats and run > best_run:
                best_run, best_start = run, run_start
        else:
            run_start, run = start, 1
        prev = key
    return best_start


def truncate_looping_output(text, min_chars=LOOPING_MIN_CHARS,
                            block_repeats=LOOPING_BLOCK_REPEATS,
                            pattern_repeats=LOOPING_PATTERN_REPEATS,
                            marker=TRUNCATION_MARKER):
    """Bound a looping output: cut at the repetition onset, append a marker.

    Returns ``text`` unchanged when it does not loop, when it is empty/None,
    or when no onset can be located (never raises, never drops a normal
    output).  The marker makes the truncation explicit in the output itself;
    callers that report status should additionally expose a machine-readable
    flag (as ``infer.py`` does in its stats line).
    """
    if not text or not is_looping_output(text, min_chars=min_chars,
                                         block_repeats=block_repeats,
                                         pattern_repeats=pattern_repeats):
        return text
    onset = looping_onset(text, min_chars=min_chars,
                          block_repeats=block_repeats,
                          pattern_repeats=pattern_repeats)
    if onset is None:
        return text
    return text[:onset] + '\n' + marker + '\n'


def main(argv=None):
    """CLI entry: check raw output files for pathological repetition.

    Exit status: 0 when no file loops (or every looping file was truncated
    with --write), 1 when a file loops and was left as-is, 2 on I/O or
    decoding errors.
    """
    parser = argparse.ArgumentParser(
        description="Detect (and optionally truncate) pathological repetition "
                    "in OCR raw output files (issue #55).")
    parser.add_argument("files", nargs="+", help="raw output files to check")
    parser.add_argument("--write", action="store_true",
                        help="truncate detected looping files in place, "
                             "appending the explicit marker")
    parser.add_argument("--min-chars", type=int, default=LOOPING_MIN_CHARS)
    parser.add_argument("--block-repeats", type=int,
                        default=LOOPING_BLOCK_REPEATS)
    parser.add_argument("--pattern-repeats", type=int,
                        default=LOOPING_PATTERN_REPEATS)
    args = parser.parse_args(argv)

    status = 0
    for path in args.files:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as exc:
            print(f"{path}: error: {exc}")
            status = max(status, 2)
            continue
        if not is_looping_output(text, min_chars=args.min_chars,
                                 block_repeats=args.block_repeats,
                                 pattern_repeats=args.pattern_repeats):
            print(f"{path}: ok")
            continue
        onset = looping_onset(text, min_chars=args.min_chars,
                              block_repeats=args.block_repeats,
                              pattern_repeats=args.pattern_repeats)
        if args.write and onset is not None:
            bounded = truncate_looping_output(
                text, min_chars=args.min_chars,
                block_repeats=args.block_repeats,
                pattern_repeats=args.pattern_repeats)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(bounded)
            except OSError as exc:
                print(f"{path}: error: {exc}")
                status = max(status, 2)
                continue
            print(f"{path}: LOOPING -> truncated "
                  f"{len(text)} -> {len(bounded)} chars")
        elif onset is None:
            print(f"{path}: LOOPING (onset not locatable; left untruncated)")
            status = max(status, 1)
        else:
            print(f"{path}: LOOPING (onset at char {onset}; use --write to "
                  "truncate)")
            status = max(status, 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
