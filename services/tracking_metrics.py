"""Workbook-based supplements for manual phase tracking metrics.

These helpers read the phase tracking spreadsheets and aggregate them into
per-user, per-phase metrics that can be merged onto universal JSON metrics.
The spreadsheets capture manual authoring and edit activity that is not always
recoverable from chat logs alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_TRACKING_WORKBOOKS: dict[str, Path] = {
    "Phase 1": Path.home() / "Downloads" / "Phase 1 Tracking.xlsx",
    "Phase 2": Path.home() / "Downloads" / "Phase 2 Tracking.xlsx",
    "Phase 3": Path.home() / "Downloads" / "Phase 3 Tracking.xlsx",
}

TRACKING_COLUMNS = [
    "task",
    "prompt",
    "context_provided",
    "writer",
    "agent_type",
    "mode",
    "lines_generated",
    "lines_ai_removed",
    "lines_edited",
    "lines_plus",
    "lines_minus",
    "missing_features",
]
KNOWN_USERS = ("jacob", "jason", "nathan", "uzayr")
TRACKING_MERGE_COLUMNS = [
    "tracking_sheet_count",
    "tracking_sheet_names",
    "tracking_workbook_paths",
    "tracking_rows",
    "tracking_prompt_rows",
    "tracking_human_written_lines",
    "tracking_human_edit_lines",
    "tracking_human_deleted_lines",
    "tracking_ai_generated_lines",
    "tracking_ai_removed_lines",
    "tracking_ai_edited_lines",
    "tracking_missing_features",
    "tracking_source_available",
]


def discover_tracking_workbooks(
    workbook_paths: dict[str, str | Path] | None = None,
) -> dict[str, Path]:
    """Return the tracking workbooks that currently exist on disk."""
    candidates = workbook_paths or DEFAULT_TRACKING_WORKBOOKS
    discovered: dict[str, Path] = {}

    for phase, raw_path in candidates.items():
        path = Path(raw_path).expanduser()
        if path.exists() and path.is_file():
            discovered[phase] = path

    return discovered


def _infer_user_from_sheet_name(sheet_name: str) -> str | None:
    """Infer the tracked user from a worksheet title."""
    normalized = str(sheet_name).strip().lower()
    for user in KNOWN_USERS:
        if user in normalized:
            return user
    return None


def _should_skip_sheet(phase: str, sheet_name: str) -> bool:
    """Skip templates, legends, and cross-phase carryover tabs."""
    normalized = str(sheet_name).strip().lower()
    if normalized in {"template", "writers"}:
        return True
    if phase == "Phase 1" and (normalized.startswith("phase 2") or normalized.startswith("phase 3")):
        return True
    return False


def _find_tracking_header_row(raw_df: pd.DataFrame) -> int | None:
    """Locate the row that defines the standard tracking columns."""
    preview_rows = min(len(raw_df), 8)
    for idx in range(preview_rows):
        row = [" ".join(str(value).strip().lower().split()) for value in raw_df.iloc[idx, :12].tolist()]
        if "prompt" in row and "writer" in row and "lines generated" in row:
            return idx
    return None


def _normalize_tracking_sheet(
    raw_df: pd.DataFrame,
    phase: str,
    user_id: str,
    workbook_path: Path,
    sheet_name: str,
) -> dict[str, object] | None:
    """Convert one worksheet into aggregate tracking metrics."""
    header_idx = _find_tracking_header_row(raw_df)
    if header_idx is None:
        return None

    data = raw_df.iloc[header_idx + 1 :, :12].copy()
    data.columns = TRACKING_COLUMNS

    for col in ["lines_generated", "lines_ai_removed", "lines_edited", "lines_plus", "lines_minus", "missing_features"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    for col in ["task", "prompt", "context_provided", "writer", "agent_type", "mode"]:
        data[col] = data[col].fillna("").astype(str).str.strip()

    data["writer_norm"] = data["writer"].str.lower()
    data["prompt_present"] = data["prompt"].ne("")
    data["row_signal"] = (
        data["prompt_present"]
        | data["writer"].ne("")
        | data[["lines_generated", "lines_ai_removed", "lines_edited", "lines_plus", "lines_minus", "missing_features"]]
        .sum(axis=1)
        .gt(0)
    )
    data = data[data["row_signal"]].copy()

    if data.empty:
        return None

    numeric_signal_total = data[
        ["lines_generated", "lines_ai_removed", "lines_edited", "lines_plus", "lines_minus", "missing_features"]
    ].sum().sum()
    if not data["writer"].ne("").any() and numeric_signal_total <= 0:
        return None

    human_rows = data[data["writer_norm"] == "human"].copy()
    model_rows = data[(data["writer_norm"] != "") & (data["writer_norm"] != "human")].copy()

    # Human-authored code is best approximated by explicit generated lines when
    # present, otherwise the manually tracked line additions.
    human_written_lines = human_rows[["lines_generated", "lines_plus"]].max(axis=1).sum()

    return {
        "phase": phase,
        "user_id": user_id,
        "tracking_workbook_path": str(workbook_path),
        "tracking_sheet_name": str(sheet_name),
        "tracking_rows": int(len(data)),
        "tracking_prompt_rows": int(data["prompt_present"].sum()),
        "tracking_human_written_lines": float(human_written_lines),
        "tracking_human_edit_lines": float(human_rows["lines_edited"].sum()),
        "tracking_human_deleted_lines": float(human_rows["lines_minus"].sum()),
        "tracking_ai_generated_lines": float(model_rows["lines_generated"].sum()),
        "tracking_ai_removed_lines": float(data["lines_ai_removed"].sum()),
        "tracking_ai_edited_lines": float(model_rows["lines_edited"].sum()),
        "tracking_missing_features": float(data["missing_features"].sum()),
    }


def load_tracking_metrics(
    workbook_paths: dict[str, str | Path] | None = None,
) -> pd.DataFrame:
    """Load and aggregate spreadsheet tracking metrics into one row per user/phase."""
    discovered = discover_tracking_workbooks(workbook_paths)
    if not discovered:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for phase, workbook_path in discovered.items():
        try:
            workbook = pd.ExcelFile(workbook_path)
        except Exception:
            continue

        for sheet_name in workbook.sheet_names:
            if _should_skip_sheet(phase, sheet_name):
                continue

            user_id = _infer_user_from_sheet_name(sheet_name)
            if not user_id:
                continue

            try:
                raw_df = workbook.parse(sheet_name, header=None)
            except Exception:
                continue

            normalized = _normalize_tracking_sheet(
                raw_df=raw_df,
                phase=phase,
                user_id=user_id,
                workbook_path=workbook_path,
                sheet_name=sheet_name,
            )
            if normalized:
                rows.append(normalized)

    if not rows:
        return pd.DataFrame()

    tracking_df = pd.DataFrame(rows)
    aggregated = (
        tracking_df.groupby(["phase", "user_id"], dropna=False)
        .agg(
            tracking_sheet_count=("tracking_sheet_name", "nunique"),
            tracking_sheet_names=("tracking_sheet_name", lambda vals: ", ".join(sorted({str(v) for v in vals}))),
            tracking_workbook_paths=(
                "tracking_workbook_path",
                lambda vals: " | ".join(sorted({str(v) for v in vals})),
            ),
            tracking_rows=("tracking_rows", "sum"),
            tracking_prompt_rows=("tracking_prompt_rows", "sum"),
            tracking_human_written_lines=("tracking_human_written_lines", "sum"),
            tracking_human_edit_lines=("tracking_human_edit_lines", "sum"),
            tracking_human_deleted_lines=("tracking_human_deleted_lines", "sum"),
            tracking_ai_generated_lines=("tracking_ai_generated_lines", "sum"),
            tracking_ai_removed_lines=("tracking_ai_removed_lines", "sum"),
            tracking_ai_edited_lines=("tracking_ai_edited_lines", "sum"),
            tracking_missing_features=("tracking_missing_features", "sum"),
        )
        .reset_index()
    )
    aggregated["tracking_source_available"] = True
    return aggregated


def merge_tracking_metrics(
    df_universal_metrics: pd.DataFrame,
    df_tracking_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Merge spreadsheet tracking metrics onto universal per-user/per-phase rows."""
    if df_universal_metrics.empty:
        return df_universal_metrics.copy()

    merged = df_universal_metrics.copy()
    if df_tracking_metrics.empty:
        for col in TRACKING_MERGE_COLUMNS:
            if col not in merged.columns:
                merged[col] = pd.NA
        return merged

    merged = merged.merge(
        df_tracking_metrics,
        on=["phase", "user_id"],
        how="left",
    )

    if "total_lines_written_by_humans" in merged.columns and "tracking_human_written_lines" in merged.columns:
        merged["total_lines_written_by_humans"] = merged["total_lines_written_by_humans"].where(
            merged["total_lines_written_by_humans"].notna(),
            merged["tracking_human_written_lines"],
        )

    return merged
