"""RITM PDF parsing helpers.

The Michelin RITM PDF is used as the source of truth for requirement
adherence scores and documented quality notes per phase.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


DEFAULT_RITM_PDF_CANDIDATES = [
    Path.home() / "Downloads" / "RITM (1).pdf",
    Path.home() / "Downloads" / "RITM.pdf",
]
RITM_COMPONENTS = [
    "Frontend",
    "Backend",
    "Database",
    "Login",
    "Navigation",
    "Common Layout",
    "Landing Page",
    "Dashboard",
    "Audit Log",
    "Settings",
    "404 Page",
]


def discover_ritm_pdf(candidates: list[str | Path] | None = None) -> Path | None:
    """Return the first local RITM PDF that exists."""
    for raw_path in candidates or DEFAULT_RITM_PDF_CANDIDATES:
        path = Path(raw_path).expanduser()
        if path.exists() and path.is_file():
            return path
    return None


def _extract_phase_blocks(reader: PdfReader) -> dict[str, str]:
    """Group PDF page text into one combined text block per phase."""
    phase_blocks: dict[str, list[str]] = {}
    current_phase: str | None = None

    for page in reader.pages:
        text = page.extract_text() or ""
        phase_match = re.search(r"Phase\s*([123])", text, re.IGNORECASE)
        if phase_match:
            current_phase = f"Phase {phase_match.group(1)}"
            phase_blocks.setdefault(current_phase, [])

        if current_phase:
            phase_blocks[current_phase].append(text)

    return {phase: "\n".join(parts) for phase, parts in phase_blocks.items()}


def _extract_phase_notes(phase_text: str, phase: str) -> list[dict[str, object]]:
    """Extract documented component notes from one phase block."""
    lines = [" ".join(line.split()) for line in phase_text.replace("\r", "\n").splitlines()]
    lines = [line for line in lines if line]

    current_component: str | None = None
    collecting_note = False
    note_lines: list[str] = []
    rows: list[dict[str, object]] = []

    def flush_note() -> None:
        nonlocal note_lines
        if current_component and note_lines:
            rows.append(
                {
                    "phase": phase,
                    "component": current_component,
                    "note": " ".join(note_lines).strip(),
                }
            )
            note_lines = []

    for line in lines:
        if line in RITM_COMPONENTS:
            flush_note()
            current_component = line
            collecting_note = False
            continue

        if line.startswith("Notes:"):
            collecting_note = True
            note_lines = [line.replace("Notes:", "", 1).strip()]
            continue

        if re.match(r"^(Total|Earned|Score):", line):
            flush_note()
            collecting_note = False
            continue

        if collecting_note:
            if line in {"0 1 2", "0 1 2 0 1 2"}:
                continue
            note_lines.append(line)

    flush_note()
    return rows


def load_ritm_data(
    pdf_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the RITM PDF into phase summary and component-note dataframes."""
    resolved_path = Path(pdf_path).expanduser() if pdf_path else discover_ritm_pdf()
    if not resolved_path or not resolved_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    try:
        reader = PdfReader(str(resolved_path))
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    phase_blocks = _extract_phase_blocks(reader)
    if not phase_blocks:
        return pd.DataFrame(), pd.DataFrame()

    phase_rows: list[dict[str, object]] = []
    note_rows: list[dict[str, object]] = []

    for phase, phase_text in phase_blocks.items():
        total_match = re.search(r"Total:\s*(\d+)", phase_text, re.IGNORECASE)
        earned_match = re.search(r"Earned:\s*(\d+)", phase_text, re.IGNORECASE)
        score_match = re.search(r"Score:\s*([\d.]+)%", phase_text, re.IGNORECASE)

        total_points = int(total_match.group(1)) if total_match else 0
        earned_points = int(earned_match.group(1)) if earned_match else 0
        score_percent = float(score_match.group(1)) if score_match else 0.0

        phase_notes = _extract_phase_notes(phase_text, phase)
        note_rows.extend(phase_notes)

        phase_rows.append(
            {
                "phase": phase,
                "total_points": total_points,
                "earned_points": earned_points,
                "missing_points": max(total_points - earned_points, 0),
                "requirement_adherence_score": score_percent / 100.0,
                "requirement_adherence_percent": score_percent,
                "requirement_count": int(total_points / 2) if total_points else None,
                "documented_issue_count": len(phase_notes),
                "components_with_documented_issues": ", ".join(
                    [row["component"] for row in phase_notes]
                ),
                "source_pdf_path": str(resolved_path),
            }
        )

    return pd.DataFrame(phase_rows), pd.DataFrame(note_rows)
