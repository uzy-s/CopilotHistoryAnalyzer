# User Guide Addendum

## 1. Guide Purpose
This guide explains how to sign in, load data, and use each tab in Copilot History Analyzer.

Audience:
- Non-technical users
- Stakeholders reviewing results
- New team members

## 2. Before You Begin
You need:
- Access to the running Streamlit app URL (typically local host URL)
- The app password from your project team
- At least one valid Copilot chat export JSON file
- Optional: a local Git repository path for commit correlation

## 3. Sign In
1. Open the app URL in your browser.
2. Enter the project password.
3. If the password is correct, the main dashboard appears.

If sign-in fails:
- Re-enter password carefully.
- Confirm APP_PASSWORD is configured.

Visual cue:
- Screenshot U1: Sign-in page showing password input

## 4. Load Your Data
Use the sidebar under Configuration.

Option A: Select Existing Phase Data
1. Select one or more phase folders.
2. The app scans those folders for JSON files.

Option B: Upload Files Manually
1. Use Or Upload chatTemplate.json manually.
2. Select one or more JSON files from your computer.

You can use both options together.

Visual cues:
- Screenshot U2: Sidebar with phase selection highlighted
- Screenshot U3: File uploader with multiple files selected

## 5. Optional Git Correlation
1. In Local Git Repository Path (Optional), paste your local repo root path.
2. If valid, commit metrics appear in statistics and timeline views.

If Git charts do not appear:
- Verify the path points to the repo root.
- Confirm the repository has commit history.

Visual cue:
- Screenshot U4: Sidebar git path field with example path

## 6. Use the Tabs
The app has five tabs:

### 6.1 Chat History
1. Choose a session under Chat History View.
2. Review prompts, responses, model, and token details.

Visual cue:
- Screenshot U5: Chat transcript with one user and assistant exchange highlighted

### 6.2 Statistics
1. Under Analysis Filters, choose sessions to include.
2. Review key metrics (code lines, tokens, success estimate, flagged reverts).
3. If git path is set, compare AI suggestion volume to git insertions.

Visual cues:
- Screenshot U6: AI Contribution and Quality panel
- Screenshot U7: Git Human Contribution panel

### 6.3 Development Timeline
1. Review scatter, line, and daily bar charts for timing and volume trends.

Visual cue:
- Screenshot U8: Timeline scatter chart and daily activity chart

### 6.4 Comparative Analysis
1. Select a baseline phase and a comparison phase.
2. Review KPI cards, side-by-side summary table, and phase distribution charts.
3. Use troubleshooting prompts table for qualitative review.

Visual cues:
- Screenshot U9: Baseline/comparison selectors and KPI cards
- Screenshot U10: Descriptor rate and model usage comparison charts

### 6.5 Prompt Analysis
1. Review top metrics (total prompts, average word count, complexity).
3. Browse the prompt repository table.
4. Review style descriptor distribution chart.

Visual cue:
- Screenshot U11: Prompt repository table with style columns visible

## 7. Typical User Workflows
### Workflow A: Quick Session Review
1. Sign in.
2. Load one phase or upload one file.
3. Select a session in Chat History.
4. Review transcript and metadata.

### Workflow B: Team Progress Comparison
1. Load at least two phases.
2. Open Comparative Analysis.
3. Select baseline and comparison phases.
4. Capture KPI and chart screenshots.

### Workflow C: AI vs Human Activity Check
1. Load chat data.
2. Enter a valid git repository path.
3. Review Statistics and Development Timeline.

## 8. Troubleshooting
Problem: No valid chat requests found
- Cause: file does not contain a top-level requests array.
- Fix: use a valid Copilot export JSON.

Problem: Incorrect password
- Cause: password mismatch.
- Fix: re-enter credentials or verify configured password.

Problem: No git statistics shown
- Cause: invalid repo path or no commits.
- Fix: verify repo root path and commit history.

Problem: Empty charts
- Cause: filters exclude all sessions.
- Fix: select at least one session in Analysis Filters.

## 9. Good Practices for End Users
- Start with one known-good JSON file to confirm parsing.
- Add more sessions gradually for easier validation.
- Use comparative mode only when at least two phases are loaded.
- Capture screenshots after filters are finalized.

## 10. Accessibility and Clarity Notes
- Read chart titles before interpretation.
- Use hover tooltips for additional chart context.
- Confirm selected sessions and phases before drawing conclusions.

## 11. Report Integration Checklist
Before final report submission, include:
- Screenshot set U1 through U11
- Short captions for each screenshot
- Brief interpretation notes for each major chart
- Any known data caveats (for example, heuristic error rate)
