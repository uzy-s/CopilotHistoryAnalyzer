"""Authentication helpers for Streamlit password gate."""

import hmac
import os

import streamlit as st


def require_password() -> None:
    """Enforce password authentication before rendering the app.

    Password lookup priority:
    1. Streamlit secrets key APP_PASSWORD.
    2. Environment variable APP_PASSWORD.

    Behavior:
    - If password config is missing, shows an error and stops execution.
    - If session is not authenticated, renders a password prompt and stops.
    - If already authenticated in session state, returns immediately.
    """
    expected_password = None
    try:
        expected_password = st.secrets.get("APP_PASSWORD")
    except Exception:
        expected_password = None

    if not expected_password:
        expected_password = os.getenv("APP_PASSWORD")

    if not expected_password:
        st.error("App password is not configured. Set APP_PASSWORD in Streamlit secrets or as an environment variable.")
        st.stop()

    def on_password_submit() -> None:
        """Validate entered password and clear plaintext input state.

        Uses constant-time comparison to reduce timing leak risk.
        """
        entered_password = st.session_state.get("app_password_input", "")
        st.session_state["password_ok"] = hmac.compare_digest(entered_password, expected_password)
        st.session_state["app_password_input"] = ""

    if st.session_state.get("password_ok", False):
        return

    st.title("Copilot History Analyzer")
    st.subheader("Sign in")
    st.text_input(
        "Password",
        type="password",
        key="app_password_input",
        on_change=on_password_submit,
    )

    if "password_ok" in st.session_state and not st.session_state["password_ok"]:
        st.error("Incorrect password")

    st.stop()
