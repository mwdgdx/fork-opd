"""Math answer verification, adapted from the SDPO repo's reward scorer.

Extracts the last \\boxed{...} from a generation and checks it against the ground
truth with the `math_verify` library (symbolic + numeric equivalence).
"""

from __future__ import annotations

import signal
from typing import Optional

from math_verify import parse as mv_parse, verify as mv_verify


def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the last ``\\boxed{...}`` substring, or None if absent."""
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None
    i = idx
    depth = 0
    right = None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return string[idx : right + 1] if right is not None else None


def remove_boxed(s: str) -> str:
    left = r"\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    return ""


class _timeout:
    def __init__(self, seconds: int = 5):
        self.seconds = seconds

    def _handle(self, signum, frame):
        raise TimeoutError

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle)
        signal.alarm(self.seconds)

    def __exit__(self, *exc):
        signal.alarm(0)


def is_correct(generation: str, ground_truth: str, timeout_s: int = 5) -> bool:
    """True iff the boxed answer in ``generation`` matches ``ground_truth``.

    ``ground_truth`` may be a bare answer or itself contain ``\\boxed{...}``.
    """
    boxed = last_boxed_only_string(generation)
    if boxed is None:
        return False
    pred = remove_boxed(boxed) or boxed

    gt_boxed = last_boxed_only_string(ground_truth)
    gt = remove_boxed(gt_boxed) if gt_boxed else ground_truth

    try:
        with _timeout(timeout_s):
            return bool(mv_verify(mv_parse(gt), mv_parse(pred)))
    except (TimeoutError, Exception):
        # fall back to exact string match on failure/timeout
        return pred.strip() == gt.strip()
