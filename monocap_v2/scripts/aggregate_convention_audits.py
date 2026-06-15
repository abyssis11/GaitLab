#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_OUT = Path("monocap_v2/benchmarks/cam0_cam1_walking_convention_summary")
BACKENDS = ["wham", "metrabs", "rtmw3d"]
RANKINGS = ["best_raw", "best_rigid", "best_pa"]
METRIC_BY_RANKING = {
    "best_raw": "raw_primary_mm",
    "best_rigid": "rigid_mm",
    "best_pa": "pa_mm",
}


def main() -> int:
    args = parse_args()
    top_paths = _resolve_top_paths(args.audit_dirs)
    rows = _dedupe_rows(_load_rows(top_paths))
    if not rows:
        raise SystemExit("No top_by_case rows found.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    best_pa = [row for row in rows if row["ranking"] == "best_pa"]
    aggregate_rows = _aggregate(best_pa)
    outputs = {
        "combined_top_by_case": out_dir / "combined_top_by_case.csv",
        "best_pa_by_case": out_dir / "best_pa_by_case.csv",
        "aggregate_by_backend_camera_hz": out_dir / "aggregate_by_backend_camera_hz.csv",
        "summary_md": out_dir / "summary.md",
    }
    _write_csv(outputs["combined_top_by_case"], rows)
    _write_csv(outputs["best_pa_by_case"], best_pa)
    _write_csv(outputs["aggregate_by_backend_camera_hz"], aggregate_rows)
    _write_summary(outputs["summary_md"], best_pa, aggregate_rows, top_paths)
    _write_metric_plot(plots_dir / "pa_mpjpe_by_case.png", best_pa, "pa_mm", "Best PA-MPJPE by Case")
    _write_metric_plot(plots_dir / "rigid_mpjpe_by_case.png", best_pa, "rigid_mm", "Rigid MPJPE at Best-PA Candidate")
    _write_metric_plot(plots_dir / "raw_root_mpjpe_by_case.png", best_pa, "raw_primary_mm", "Raw Root-Centered MPJPE at Best-PA Candidate")
    _write_offset_plot(plots_dir / "time_offset_by_case.png", best_pa)

    print(f"[aggregate-convention-audits] cases={len(best_pa)} top_rows={len(rows)}")
    print(f"[aggregate-convention-audits] summary -> {outputs['summary_md']}")
    print(f"[aggregate-convention-audits] plots -> {plots_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate per-case backend convention audit top rows.")
    ap.add_argument(
        "--audit-dirs",
        default="",
        help="Comma-separated audit directories. Defaults to cam*_walking*_convention_audit*/top_by_case.csv under benchmarks.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def _resolve_top_paths(value: str) -> list[Path]:
    if value.strip():
        candidates = [Path(item.strip()) for item in value.split(",") if item.strip()]
        return [path / "top_by_case.csv" if path.is_dir() else path for path in candidates]
    root = Path("monocap_v2/benchmarks")
    return sorted(root.glob("cam*_walking*_convention_audit*/top_by_case.csv"))


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["source"] = str(path)
                rows.append(_coerce_row(row))
    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, float, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["camera"], row["trial"], row["backend"], float(row["evaluation_hz"]), row["ranking"])
        current = by_key.get(key)
        metric = METRIC_BY_RANKING.get(row["ranking"], "metric_value_mm")
        if current is None or _float(row.get(metric)) < _float(current.get(metric)):
            by_key[key] = row
    return sorted(by_key.values(), key=lambda r: (_camera_key(r["camera"]), r["trial"], r["backend"], float(r["evaluation_hz"]), r["ranking"]))


def _aggregate(best_pa: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in best_pa:
        groups.setdefault((row["camera"], row["backend"], float(row["evaluation_hz"])), []).append(row)

    out = []
    for (camera, backend, hz), group in sorted(groups.items(), key=lambda item: (_camera_key(item[0][0]), item[0][1], item[0][2])):
        pa = [_float(row["pa_mm"]) for row in group]
        rigid = [_float(row["rigid_mm"]) for row in group]
        raw = [_float(row["raw_primary_mm"]) for row in group]
        offsets = [_float(row["time_offset_s"]) for row in group]
        out.append(
            {
                "camera": camera,
                "backend": backend,
                "evaluation_hz": hz,
                "trial_count": len(group),
                "median_pa_mm": _median(pa),
                "mean_pa_mm": _mean(pa),
                "median_rigid_mm": _median(rigid),
                "median_raw_root_mm": _median(raw),
                "median_time_offset_s": _median(offsets),
                "axes": ";".join(sorted({str(row["axis"]) for row in group})),
                "diagnostic_only_count": sum(1 for row in group if _bool(row.get("diagnostic_only"))),
            }
        )
    return out


def _write_summary(path: Path, best_pa: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], top_paths: list[Path]) -> None:
    lines = [
        "# Cam0/Cam1 Walking Convention Audit Summary",
        "",
        f"- Source audit files: `{len([p for p in top_paths if p.exists()])}`",
        f"- Best-PA case rows: `{len(best_pa)}`",
        "- Metrics are millimeters.",
        "- Rows are diagnostic audit results; they do not promote a production convention.",
        "",
        "## Aggregate Median PA-MPJPE",
        "",
    ]
    for hz in sorted({float(row["evaluation_hz"]) for row in aggregate_rows}):
        lines.extend([f"### {hz:g} Hz", ""])
        rows = [row for row in aggregate_rows if float(row["evaluation_hz"]) == hz]
        lines.extend(_markdown_table(["Camera", "Backend", "Trials", "Median PA", "Median Rigid", "Median Raw", "Median Offset", "Axes"], [
            [
                row["camera"],
                row["backend"],
                row["trial_count"],
                _fmt(row["median_pa_mm"]),
                _fmt(row["median_rigid_mm"]),
                _fmt(row["median_raw_root_mm"]),
                _fmt(row["median_time_offset_s"], digits=3),
                f"`{row['axes']}`",
            ]
            for row in rows
        ]))
        lines.append("")

    for hz in sorted({float(row["evaluation_hz"]) for row in best_pa}):
        lines.extend([f"## Best PA-MPJPE By Case, {hz:g} Hz", ""])
        rows = [row for row in best_pa if float(row["evaluation_hz"]) == hz]
        lines.extend(_case_metric_table(rows, "pa_mm"))
        lines.append("")
        lines.extend([f"## Timing Offset At Best PA, {hz:g} Hz", ""])
        lines.extend(_case_metric_table(rows, "time_offset_s", digits=3))
        lines.append("")
        lines.extend([f"## Axis At Best PA, {hz:g} Hz", ""])
        lines.extend(_case_axis_table(rows))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_metric_table(rows: list[dict[str, Any]], metric: str, digits: int = 2) -> list[str]:
    cases = sorted({(row["camera"], row["trial"]) for row in rows}, key=lambda item: (_camera_key(item[0]), item[1]))
    lookup = {(row["camera"], row["trial"], row["backend"]): row for row in rows}
    return _markdown_table(["Camera", "Trial", *BACKENDS], [
        [camera, trial, *[_fmt(_float((lookup.get((camera, trial, backend)) or {}).get(metric)), digits=digits) for backend in BACKENDS]]
        for camera, trial in cases
    ])


def _case_axis_table(rows: list[dict[str, Any]]) -> list[str]:
    cases = sorted({(row["camera"], row["trial"]) for row in rows}, key=lambda item: (_camera_key(item[0]), item[1]))
    lookup = {(row["camera"], row["trial"], row["backend"]): row for row in rows}
    table_rows = []
    for camera, trial in cases:
        values = []
        for backend in BACKENDS:
            row = lookup.get((camera, trial, backend)) or {}
            axis = row.get("axis")
            values.append(f"`{axis}`" if axis else "")
        table_rows.append([camera, trial, *values])
    return _markdown_table(["Camera", "Trial", *BACKENDS], table_rows)


def _write_metric_plot(path: Path, rows: list[dict[str, Any]], metric: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cases = sorted({(row["camera"], row["trial"], float(row["evaluation_hz"])) for row in rows}, key=lambda item: (_camera_key(item[0]), item[1], item[2]))
    lookup = {(row["camera"], row["trial"], float(row["evaluation_hz"]), row["backend"]): row for row in rows}
    labels = [f"{camera}\n{trial}\n{hz:g}Hz" for camera, trial, hz in cases]
    x = np.arange(len(cases))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(12, len(cases) * 0.8), 5.8))
    colors = {"wham": "#4C78A8", "metrabs": "#F58518", "rtmw3d": "#54A24B"}
    for idx, backend in enumerate(BACKENDS):
        values = [_float((lookup.get((camera, trial, hz, backend)) or {}).get(metric)) for camera, trial, hz in cases]
        ax.bar(x + (idx - 1) * width, values, width, label=backend, color=colors.get(backend))
    ax.set_title(title)
    ax.set_ylabel("mm")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_offset_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cases = sorted({(row["camera"], row["trial"], float(row["evaluation_hz"])) for row in rows}, key=lambda item: (_camera_key(item[0]), item[1], item[2]))
    lookup = {(row["camera"], row["trial"], float(row["evaluation_hz"]), row["backend"]): row for row in rows}
    labels = [f"{camera}\n{trial}\n{hz:g}Hz" for camera, trial, hz in cases]
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(max(12, len(cases) * 0.8), 5.6))
    markers = {"wham": "o", "metrabs": "s", "rtmw3d": "^"}
    for backend in BACKENDS:
        values = [_float((lookup.get((camera, trial, hz, backend)) or {}).get("time_offset_s")) for camera, trial, hz in cases]
        ax.plot(x, values, marker=markers.get(backend, "o"), label=backend)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Timing Offset at Best PA Candidate")
    ax.set_ylabel("seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return out


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in [
        "evaluation_hz",
        "metric_value_mm",
        "raw_primary_mm",
        "rigid_mm",
        "pa_mm",
        "axis_determinant",
        "time_offset_s",
        "overlap_frames",
        "time_start_s",
        "time_end_s",
    ]:
        if key in row:
            row[key] = _float(row[key])
    return row


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _median(values: list[float]) -> float:
    clean = [value for value in values if value == value]
    return float(statistics.median(clean)) if clean else float("nan")


def _mean(values: list[float]) -> float:
    clean = [value for value in values if value == value]
    return float(statistics.fmean(clean)) if clean else float("nan")


def _fmt(value: Any, digits: int = 2) -> str:
    number = _float(value)
    if number != number:
        return ""
    return f"{number:.{digits}f}"


def _camera_key(value: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return (int(digits) if digits else 999, str(value))


if __name__ == "__main__":
    raise SystemExit(main())
