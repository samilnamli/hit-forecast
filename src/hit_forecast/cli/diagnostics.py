"""Phase-0 diagnostics over one or more cache shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..features.cache import load_cache
from ..models import combine_caches
from ..train import signal_diagnostics
from ..utils import get_logger

_log = get_logger("diagnostics")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compute go/no-go routing-signal diagnostics")
    ap.add_argument("caches", nargs="+", help="cache shard directories")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    caches = [load_cache(c) for c in args.caches]
    data = combine_caches(caches)
    diag = signal_diagnostics(data.mase, data.expert_names)
    print(json.dumps(diag, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(diag, indent=2))
    if not diag["gate_pass"]:
        _log.warning("GATE FAILED: routing signal is weak for this pool/corpus. "
                     "Swap in more diverse / less contaminated experts before training.")


if __name__ == "__main__":
    main()
