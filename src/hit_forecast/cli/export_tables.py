"""Export paper-ready booktabs tables from metrics.csv + diagnostics.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from ..utils.logging import get_logger

_log = get_logger("export_tables")

# Default methods for the main GiftEval table (clean pool).
_MAIN_METHODS = [
    "single:chronos-2",
    "single:moirai-base",
    "single:timesfm-2.5",
    "single:tirex",
    "single:theta",
    "single:snaive",
    "single:drift",
    "ensemble:weighted",
    "pooled_mlp:hard",
    "hit_forecast:hard",
    "hit_forecast:soft",
    "oracle",
]

_CONTAM_METHODS = [
    "single:chronos-t5-small",
    "single:chronos-bolt-base",
    "single:chronos-t5-base",
    "single:moirai-base",
    "ensemble:weighted",
    "pooled_mlp:hard",
    "hit_forecast:hard",
    "oracle",
]

_DEFAULT_EXCLUDE = ("bitbrains",)


def _short_ds(name: str) -> str:
    name = name.replace("::test", "").replace("::train", "")
    name = name.split("/")[0]
    return name.replace("_with_missing", "").replace("_", r"\_")


def _short_method(m: str) -> str:
    return m.replace("single:", "").replace("ensemble:", "ens:").replace("_", r"\_")


def load_metrics_csv(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """dataset -> method -> metric -> value."""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            ds, method = row["dataset"], row["method"]
            mets = {}
            for k, v in row.items():
                if k in ("dataset", "method") or v in ("", None):
                    continue
                try:
                    mets[k] = float(v)
                except ValueError:
                    continue
            out[ds][method] = mets
    return dict(out)


def _excluded(ds: str, exclude: tuple[str, ...]) -> bool:
    return any(s in ds for s in exclude)


def macro_gmean(
    data: dict[str, dict[str, dict[str, float]]],
    method: str,
    metric: str = "MASE",
    exclude: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[float | None, float | None, int]:
    vals = []
    for ds, methods in data.items():
        if _excluded(ds, exclude):
            continue
        v = methods.get(method, {}).get(metric)
        if v is not None and v > 0 and math.isfinite(v):
            vals.append(v)
    if not vals:
        return None, None, 0
    macro = sum(vals) / len(vals)
    gmean = math.exp(sum(math.log(v) for v in vals) / len(vals))
    return macro, gmean, len(vals)


def booktabs_main(
    data: dict[str, dict[str, dict[str, float]]],
    methods: list[str],
    caption: str,
    label: str,
    exclude: tuple[str, ...] = _DEFAULT_EXCLUDE,
    metric: str = "MASE",
) -> str:
    # Keep methods that appear at least once.
    methods = [m for m in methods if any(m in data[ds] for ds in data)]
    header = "Dataset & " + " & ".join(_short_method(m) for m in methods) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\begin{tabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    datasets = sorted(ds for ds in data if not _excluded(ds, exclude))
    for ds in datasets:
        cells = [_short_ds(ds)]
        for m in methods:
            v = data[ds].get(m, {}).get(metric)
            cells.append(f"{v:.3f}" if v is not None and math.isfinite(v) else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    # macro / gmean rows
    for row_name, kind in (("Macro mean", "macro"), (r"Geo.\ mean", "gmean")):
        cells = [row_name]
        for m in methods:
            mac, gm, _ = macro_gmean(data, m, metric=metric, exclude=exclude)
            val = mac if kind == "macro" else gm
            cells.append(f"{val:.3f}" if val is not None else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def booktabs_phase0(clean: dict, contam: dict, caption: str, label: str) -> str:
    keys = [
        ("n_windows", "Train windows"),
        ("win_rate_entropy_norm", "Win-rate entropy (norm.)"),
        ("rel_margin_median", r"Median rel.\ MASE margin"),
        ("abs_margin_median", r"Median abs.\ MASE margin"),
        ("oracle_gain_vs_best_single", "Oracle gain vs best single"),
        ("gate_pass", "Phase-0 gate"),
    ]

    def fmt(v):
        if isinstance(v, bool):
            return "pass" if v else "fail"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Clean pool & Contaminated pool \\",
        r"\midrule",
    ]
    for key, name in keys:
        lines.append(f"{name} & {fmt(clean.get(key))} & {fmt(contam.get(key))} \\\\")
    # win rates as a compact note
    lines.append(r"\midrule")
    cw = clean.get("win_rate") or {}
    tw = contam.get("win_rate") or {}
    lines.append(
        r"\multicolumn{3}{l}{\scriptsize Clean wins: "
        + ", ".join(f"{k}={v:.2f}" for k, v in cw.items())
        + r"} \\"
    )
    lines.append(
        r"\multicolumn{3}{l}{\scriptsize Contam.\ wins: "
        + ", ".join(f"{k}={v:.2f}" for k, v in tw.items())
        + r"} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def booktabs_seeds(
    seed_dirs: list[Path],
    methods: list[str],
    caption: str,
    label: str,
    exclude: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> str:
    """Mean ± std of geo-mean MASE across seeds."""
    per_seed: list[dict[str, float]] = []
    for d in seed_dirs:
        csv_path = d / "metrics.csv"
        if not csv_path.exists():
            continue
        data = load_metrics_csv(csv_path)
        row = {}
        for m in methods:
            _, gm, _ = macro_gmean(data, m, exclude=exclude)
            if gm is not None:
                row[m] = gm
        if row:
            per_seed.append(row)
    methods = [m for m in methods if any(m in r for r in per_seed)]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Method & Geo.\ mean MASE (mean$\pm$std) \\",
        r"\midrule",
    ]
    for m in methods:
        vals = [r[m] for r in per_seed if m in r]
        if not vals:
            continue
        mu = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
            sd = math.sqrt(var)
            cell = f"{mu:.3f}$\\pm${sd:.3f}"
        else:
            cell = f"{mu:.3f}"
        lines.append(f"{_short_method(m)} & {cell} \\\\")
    lines.extend([
        rf"\multicolumn{{2}}{{l}}{{\scriptsize $n={len(per_seed)}$ seeds; bitbrains excluded.}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def summary_json(
    clean_data: dict | None,
    contam_data: dict | None,
    clean_diag: dict | None,
    contam_diag: dict | None,
    exclude: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> dict:
    out: dict = {"exclude": list(exclude)}
    for tag, data, methods in (
        ("clean", clean_data, _MAIN_METHODS),
        ("contaminated", contam_data, _CONTAM_METHODS),
    ):
        if not data:
            continue
        block = {}
        for m in methods:
            mac, gm, n = macro_gmean(data, m, exclude=exclude)
            if n:
                block[m] = {"macro_mase": mac, "gmean_mase": gm, "n": n}
        out[tag] = block
    if clean_diag:
        out["diagnostics_clean"] = clean_diag
    if contam_diag:
        out["diagnostics_contaminated"] = contam_diag
    return out


def _find_seed_dirs(root: Path) -> list[Path]:
    """Prefer results/gifteval_main_clean/seed_* ; also accept root itself as seed_0."""
    seeds = sorted(root.glob("seed_*"))
    if seeds:
        return [p for p in seeds if (p / "metrics.csv").exists()]
    if (root / "metrics.csv").exists():
        return [root]
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export paper booktabs tables")
    ap.add_argument(
        "--clean",
        type=Path,
        default=Path("results/gifteval_main_clean"),
        help="Clean-pool results directory (metrics.csv, diagnostics.json)",
    )
    ap.add_argument(
        "--contaminated",
        type=Path,
        default=Path("results/gifteval_contaminated"),
        help="Contaminated-pool results directory",
    )
    ap.add_argument(
        "--ablation-loss",
        type=Path,
        default=Path("results/ablation_loss_mase"),
    )
    ap.add_argument(
        "--ablation-arch",
        type=Path,
        default=Path("results/ablation_no_ca"),
    )
    ap.add_argument("--out", type=Path, default=Path("results/paper"))
    ap.add_argument(
        "--exclude",
        default="bitbrains",
        help="Comma-separated substrings of dataset names to drop from macros",
    )
    args = ap.parse_args(argv)
    exclude = tuple(s.strip() for s in args.exclude.split(",") if s.strip())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    clean_data = load_metrics_csv(args.clean / "metrics.csv") if (args.clean / "metrics.csv").exists() else None
    contam_data = (
        load_metrics_csv(args.contaminated / "metrics.csv")
        if (args.contaminated / "metrics.csv").exists()
        else None
    )
    clean_diag = (
        json.loads((args.clean / "diagnostics.json").read_text())
        if (args.clean / "diagnostics.json").exists()
        else None
    )
    contam_diag = (
        json.loads((args.contaminated / "diagnostics.json").read_text())
        if (args.contaminated / "diagnostics.json").exists()
        else None
    )

    written = []
    if clean_data:
        tex = booktabs_main(
            clean_data,
            _MAIN_METHODS,
            caption="GiftEval short (clean pool): MASE by method. Bitbrains excluded from means.",
            label="tab:gifteval-main-clean",
            exclude=exclude,
        )
        p = out / "table_main.tex"
        p.write_text(tex)
        written.append(str(p))

    if contam_data:
        tex = booktabs_main(
            contam_data,
            _CONTAM_METHODS,
            caption="GiftEval short (contaminated pool): MASE by method.",
            label="tab:gifteval-contaminated",
            exclude=exclude,
        )
        p = out / "table_contaminated.tex"
        p.write_text(tex)
        written.append(str(p))

    if clean_diag and contam_diag:
        tex = booktabs_phase0(
            clean_diag,
            contam_diag,
            caption="Phase-0 routing-signal diagnostics: clean vs contaminated expert pool.",
            label="tab:phase0-contrast",
        )
        p = out / "table_phase0.tex"
        p.write_text(tex)
        written.append(str(p))

    if args.ablation_loss.exists() and (args.ablation_loss / "metrics.csv").exists():
        data = load_metrics_csv(args.ablation_loss / "metrics.csv")
        tex = booktabs_main(
            data,
            ["pooled_mlp:hard", "hit_forecast:hard", "hit_forecast:soft", "oracle"],
            caption="Ablation: MASE-only loss ($\\lambda_{\\mathrm{hard}}=\\lambda_{\\mathrm{soft}}=0$).",
            label="tab:ablation-loss",
            exclude=exclude,
        )
        p = out / "table_ablation_loss.tex"
        p.write_text(tex)
        written.append(str(p))

    if args.ablation_arch.exists() and (args.ablation_arch / "metrics.csv").exists():
        data = load_metrics_csv(args.ablation_arch / "metrics.csv")
        tex = booktabs_main(
            data,
            ["pooled_mlp:hard", "hit_forecast:hard", "hit_forecast:soft", "oracle"],
            caption="Ablation: hierarchical router without cross-attention bridge.",
            label="tab:ablation-arch",
            exclude=exclude,
        )
        p = out / "table_ablation_arch.tex"
        p.write_text(tex)
        written.append(str(p))

    seed_dirs = _find_seed_dirs(args.clean)
    # Also collect sibling seed_* under parent if main dir is seed_0 parent layout
    if len(seed_dirs) <= 1:
        parent_seeds = sorted(args.clean.parent.glob("gifteval_main_clean_seed_*"))
        seed_dirs = [p for p in parent_seeds if (p / "metrics.csv").exists()] or seed_dirs
    if seed_dirs:
        tex = booktabs_seeds(
            seed_dirs,
            ["single:tirex", "pooled_mlp:hard", "hit_forecast:hard", "oracle"],
            caption="Clean-pool geo-mean MASE across random seeds.",
            label="tab:seeds",
            exclude=exclude,
        )
        p = out / "table_seeds.tex"
        p.write_text(tex)
        written.append(str(p))

    summary = summary_json(clean_data, contam_data, clean_diag, contam_diag, exclude)
    sp = out / "summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    written.append(str(sp))

    _log.info("Wrote %d artifacts under %s", len(written), out)
    for w in written:
        print(w)


if __name__ == "__main__":
    main()
