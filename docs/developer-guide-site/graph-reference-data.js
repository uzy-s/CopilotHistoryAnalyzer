// Graph descriptions and their underlying data sources, grouped by UI Tab

window.GRAPH_REFERENCE_DATA = [
  {
    tab: "Chat Explorer",
    description: "This tab does not contain graphs. It provides a reconstructed view of the raw chat sessions by parsing the selected JSON metadata.",
    graphs: []
  },
  {
    tab: "Statistics",
    description: "A side-by-side comparison of AI-generated code versus Human-committed code.",
    graphs: [
      {
        title: "AI Models Used",
        description: "A donut chart showing the market share of which AI models generated the responses.",
        source: "Chat JSON files (model field).",
        metric: "Count of responses per model."
      },
      {
        title: "Code Commits by Author",
        description: "A pie chart showing human code contributions broken down by Git author name.",
        source: "Local Git Repository.",
        metric: "Count of commits per author."
      },
      {
        title: "AI Suggestions vs Committed Code Volume",
        description: "A bar chart comparing the total amount of AI-suggested code lines against the total amount of lines inserted by humans into the Git repository. If the Git repository is filtered to a specific phase, the AI lines are also filtered to that phase for an apples-to-apples comparison.",
        source: "Chat JSON files (code_lines_suggested) vs Local Git Repository (insertions).",
        metric: "Code volume (lines)."
      },
      {
        title: "Total Response Latency",
        description: "A box plot showing the distribution of total time it took for the AI to respond, grouped by model.",
        source: "Chat JSON files (latency_ms).",
        metric: "Seconds."
      },
      {
        title: "Thinking Time / TTFT",
        description: "A box plot showing the Time To First Token (TTFT) distribution, grouped by model.",
        source: "Chat JSON files (ttft_ms).",
        metric: "Seconds."
      },
      {
        title: "Top Programming Languages Generated",
        description: "A bar chart showing the frequency of different programming languages detected in the AI's code fences.",
        source: "Chat JSON files (languages array).",
        metric: "Frequency count."
      },
      {
        title: "Top 10 Context Files",
        description: "A horizontal bar chart identifying which local workspace files were most frequently referenced by the AI as context.",
        source: "Chat JSON files (referenced_files array).",
        metric: "Frequency count."
      },
      {
        title: "File Activity Frequency",
        description: "A bar chart comparing the frequency of Checkpoint Restorations (undo stops) vs direct Editor Edits applied by the AI.",
        source: "Chat JSON files (checkpoints_restored vs edited_file_events).",
        metric: "Event count."
      }
    ]
  },
  {
    tab: "Executive Dashboard",
    description: "A high-level dashboard combining Universal metrics, unified overviews, and key actionable insights.",
    graphs: [
      {
        title: "Measured AI Output Lines by Phase",
        description: "A bar chart displaying the total AI lines of code generated per project phase.",
        source: "Universal Synthesis JSON.",
        metric: "Total lines written by selected model/agent."
      },
      {
        title: "Work Hours by Phase",
        description: "A bar chart showing the sum of human tracking hours recorded per phase.",
        source: "Universal Synthesis JSON & Tracking Workbooks.",
        metric: "Man hours."
      },
      {
        title: "Team Man-Days by Phase",
        description: "A bar chart showing the sum of standardized man-days recorded per phase.",
        source: "Universal Synthesis JSON & Tracking Workbooks.",
        metric: "Man days."
      },
      {
        title: "Prompt-to-Feature Ratio by Phase",
        description: "A bar chart showing the ratio of visible prompts to successfully completed heuristic feature runs.",
        source: "Universal Synthesis JSON.",
        metric: "Ratio (Lower is better)."
      },
      {
        title: "Prompt Success Rate by Phase",
        description: "A bar chart displaying the percentage of prompts that successfully contributed to the project without requiring immediate retry.",
        source: "Universal Synthesis JSON.",
        metric: "Percentage (%)."
      },
      {
        title: "Measured Output Composition by Phase",
        description: "A stacked bar chart breaking down the AI's output into 'Assistant Fallback Code Lines' vs 'Structured Edit Lines'.",
        source: "Universal Synthesis JSON.",
        metric: "Line counts."
      },
      {
        title: "Requirement Adherence Score",
        description: "A bar chart comparing the actual RITM score against the maximum possible RITM score for each phase.",
        source: "RITM PDF Report.",
        metric: "Points."
      },
      {
        title: "Universal Turn Count by User and Phase",
        description: "A grouped bar chart displaying the total number of chat interactions broken down by individual user and project phase.",
        source: "Universal Synthesis JSON.",
        metric: "Interaction count."
      },
      {
        title: "AI Output and Revision Pressure",
        description: "A grouped bar chart comparing the total AI lines generated against the number of those lines that required human revision.",
        source: "Universal Synthesis JSON.",
        metric: "Line counts."
      }
    ]
  },
  {
    tab: "Comparative Analysis & Timelines",
    description: "Cross-phase comparisons of prompt complexities, styles, flows, and time-series event metrics.",
    graphs: [
      {
        title: "Descriptor Rates by Phase (%)",
        description: "A grouped bar chart comparing the tone and styling descriptors (e.g., Inquisitive, Polite, Direct, Troubleshooting) between two selected phases.",
        source: "Chat JSON files (Analyzed Prompt Descriptors).",
        metric: "Percentage of prompts."
      },
      {
        title: "Model Usage Comparison (% of prompts)",
        description: "A grouped bar chart comparing the percentage share of different AI models used between two selected phases.",
        source: "Chat JSON files (model field).",
        metric: "Percentage of prompts."
      },
      {
        title: "Complexity Distribution by Phase",
        description: "A box plot showing the distribution and spread of prompt complexity scores across two selected phases.",
        source: "Chat JSON files (Calculated Complexity).",
        metric: "Complexity Score (1-10)."
      },
      {
        title: "Prompt Length Profile (% of prompts)",
        description: "A grouped bar chart comparing the percentage distribution of prompt lengths (Short, Medium, Long) between two selected phases.",
        source: "Chat JSON files (Calculated Word Count).",
        metric: "Percentage of prompts."
      },
      {
        title: "Daily Interaction Share by Relative Day (%)",
        description: "A line chart plotting the trajectory of user interaction share on a day-by-day basis, normalized by the start of the phase so both phases overlay each other on 'Relative Day 1, 2, 3...'",
        source: "Chat JSON files (timestamp).",
        metric: "Daily Share (%)."
      },
      {
        title: "Activity Timeline (Scatter)",
        description: "A scatter plot mapping every chat interaction and Git commit across the timeline of the project. The size of the bubble corresponds to the code volume (suggested lines or inserted lines).",
        source: "Universal Synthesis JSON / Chat JSON & Local Git Repository.",
        metric: "Code volume."
      },
      {
        title: "Code Volume Over Time",
        description: "A line chart tracking cumulative or periodic code volume bursts over the course of the project.",
        source: "Universal Synthesis JSON / Chat JSON & Local Git Repository.",
        metric: "Code volume."
      },
      {
        title: "Daily Interactions vs Commits",
        description: "A daily bar chart showing the raw total volume of AI interactions alongside the raw volume of human Git commits per day.",
        source: "Universal Synthesis JSON / Chat JSON & Local Git Repository.",
        metric: "Total count of events."
      },
      {
        title: "Prompt-to-Feature Flow",
        description: "A Sankey Diagram visualizing the funnel of raw prompts, breaking down how they convert into parsed actions, file touches, and eventually successful code output vs abandoned changes.",
        source: "Chat JSON files.",
        metric: "Flow volume."
      }
    ]
  }
];