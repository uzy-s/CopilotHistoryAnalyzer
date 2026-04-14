"""Reusable analytics helpers used by multiple Streamlit tabs."""

import pandas as pd


def calculate_success_metrics(df: pd.DataFrame) -> tuple[int, int]:
    """Estimate response success using a follow-up heuristic.

    Args:
        df: Chat dataframe containing file_name, timestamp, code_lines_suggested,
            and user_text columns.

    Returns:
        A tuple of (total_code_responses, flagged_reverts).

    Notes:
        A revert is flagged when the next user prompt is short and contains
        error/fix style keywords.
    """
    total_code_responses = 0
    flagged_reverts = 0
    negative_keywords = [
        "error",
        "fix",
        "no",
        "wrong",
        "fail",
        "broken",
        "bug",
        "issue",
        "doesn't work",
        "didn't work",
        "restore",
    ]

    unique_sessions = df["file_name"].unique()

    for session in unique_sessions:
        session_df = df[df["file_name"] == session].sort_values("timestamp")
        rows = list(session_df.iterrows())

        for i in range(len(rows) - 1):
            _, curr_row = rows[i]
            _, next_row = rows[i + 1]

            if curr_row["code_lines_suggested"] <= 0:
                continue

            total_code_responses += 1
            next_user_msg = str(next_row["user_text"]).lower()
            word_count = len(next_user_msg.split())

            if word_count < 20 and any(keyword in next_user_msg for keyword in negative_keywords):
                flagged_reverts += 1

    return total_code_responses, flagged_reverts


def analyze_prompt_style(df: pd.DataFrame) -> pd.DataFrame:
    """Classify prompt tone/style and compute complexity descriptors.

    Args:
        df: Chat dataframe containing user_text and metadata columns.

    Returns:
        A dataframe with one row per prompt and derived style flags, counts,
        and complexity fields used by analysis tabs.
    """
    results = []

    for _, row in df.iterrows():
        text = str(row.get("user_text", "")).strip()
        if not text:
            continue

        word_count = len(text.split())
        char_count = len(text)
        text_lower = text.lower()
        descriptors = []

        # Descriptor extraction is intentionally heuristic and keyword-based.
        if "?" in text or any(
            word in text_lower.split() for word in ["how", "what", "why", "where", "when", "who", "explain"]
        ):
            descriptors.append("Inquisitive")

        if any(word in text_lower for word in ["please", "thanks", "thank you", "could you", "would you"]):
            descriptors.append("Polite")

        if any(text_lower.startswith(word) for word in ["create", "make", "write", "generate", "add", "update", "fix"]):
            if word_count < 15:
                descriptors.append("Direct/Commanding")
            else:
                descriptors.append("Detailed Request")

        if any(word in text_lower for word in ["error", "bug", "fail", "broken", "issue", "doesn't work"]):
            descriptors.append("Troubleshooting")

        if not descriptors:
            if word_count < 5:
                descriptors.append("Succinct/Short")
            else:
                descriptors.append("Conversational")

        unique_words = len(set(text_lower.split()))
        complexity_score = min(10, max(1, int((unique_words / 15) + (char_count / 150))))
        prompt_length_bucket = "Short" if word_count < 12 else "Medium" if word_count < 30 else "Long"
        descriptor_set = set(descriptors)

        results.append(
            {
                "Session": row.get("file_name", "Unknown"),
                "Timestamp": row.get("timestamp", ""),
                "User": row.get("suspected_user", "Unknown"),
                "Phase": row.get("phase", "Uploaded"),
                "Prompt": text,
                "Word Count": word_count,
                "Complexity Score": complexity_score,
                "Prompt Length Bucket": prompt_length_bucket,
                "Style Descriptors": ", ".join(descriptors),
                "Is Inquisitive": "Inquisitive" in descriptor_set,
                "Is Polite": "Polite" in descriptor_set,
                "Is Direct": "Direct/Commanding" in descriptor_set,
                "Is Detailed": "Detailed Request" in descriptor_set,
                "Is Troubleshooting": "Troubleshooting" in descriptor_set,
            }
        )

    return pd.DataFrame(results)


def safe_mean(series: pd.Series) -> float:
    """Return a safe mean value for UI metrics.

    Args:
        series: Numeric pandas series.

    Returns:
        Mean value when non-empty, else 0.0.
    """
    return float(series.mean()) if len(series) > 0 else 0.0


def safe_rate(df: pd.DataFrame, col: str) -> float:
    """Compute percentage rate for a boolean indicator column.

    Args:
        df: Dataframe containing boolean indicator columns.
        col: Column name to aggregate.

    Returns:
        Percentage in the range 0-100, or 0.0 for empty dataframes.
    """
    return float(df[col].mean() * 100.0) if len(df) > 0 else 0.0


def phase_duration_days(df_phase_chat: pd.DataFrame) -> float:
    """Compute elapsed phase duration in days.

    Args:
        df_phase_chat: Dataframe containing timestamp rows for a phase.

    Returns:
        Non-negative day span between min and max timestamps.
    """
    if df_phase_chat.empty:
        return 0.0

    start_ts = df_phase_chat["timestamp"].min()
    end_ts = df_phase_chat["timestamp"].max()
    duration = end_ts - start_ts
    return max(duration.total_seconds() / 86400.0, 0.0)


def tokens_per_prompt(df_phase_chat: pd.DataFrame) -> float:
    """Compute average token volume per prompt.

    Args:
        df_phase_chat: Dataframe with prompt_tokens and completion_tokens.

    Returns:
        Average total tokens per row, or 0.0 for empty input.
    """
    if df_phase_chat.empty:
        return 0.0

    total_tokens = df_phase_chat["completion_tokens"].sum() + df_phase_chat["prompt_tokens"].sum()
    return float(total_tokens / len(df_phase_chat)) if len(df_phase_chat) > 0 else 0.0


def error_rate_percent(df_phase_chat: pd.DataFrame) -> float:
    """Compute estimated error/revert percentage from heuristic counters.

    Args:
        df_phase_chat: Phase-scoped chat dataframe.

    Returns:
        Percent of flagged reverts among code-producing responses.
    """
    total_code, flagged_reverts = calculate_success_metrics(df_phase_chat)
    if total_code <= 0:
        return 0.0
    return (flagged_reverts / total_code) * 100.0


def avg_interactions_per_day(df_phase_chat: pd.DataFrame) -> float:
    """Compute normalized interaction density.

    Args:
        df_phase_chat: Phase-scoped chat dataframe.

    Returns:
        Interactions per day, using a minimum denominator of 1 day.
    """
    if df_phase_chat.empty:
        return 0.0

    duration_days = phase_duration_days(df_phase_chat)
    normalized_days = max(duration_days, 1.0)
    return len(df_phase_chat) / normalized_days


def top_model_share_percent(df_phase_chat: pd.DataFrame) -> tuple[float, str]:
    """Return dominant model share for a phase.

    Args:
        df_phase_chat: Phase-scoped chat dataframe containing model values.

    Returns:
        Tuple of (share_percent, model_name). Returns (0.0, "N/A") when empty.
    """
    if df_phase_chat.empty:
        return 0.0, "N/A"

    model_counts = df_phase_chat["model"].value_counts()
    if model_counts.empty:
        return 0.0, "N/A"

    top_model = model_counts.index[0]
    top_share = (model_counts.iloc[0] / model_counts.sum()) * 100.0
    return top_share, top_model
