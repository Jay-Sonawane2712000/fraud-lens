from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_PATH = FIGURE_DIR / "arch_diagram.png"


def add_box(ax, x, y, width, height, label, facecolor, edgecolor="#283747"):
    """Draw a labeled box for the README/project documentation diagram."""
    box = plt.Rectangle(
        (x, y),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color="#1f2933",
        wrap=True,
        zorder=3,
    )


def add_arrow(ax, start, end, color="#34495e"):
    """Draw a clean directional arrow between pipeline boxes."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.8, color=color),
        zorder=1,
    )


def main():
    """Create the architecture diagram used in README/project documentation."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        8,
        9.45,
        "FraudLens System Architecture",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#17202a",
    )

    pipeline = [
        ("Raw\nIEEE-CIS Data", 0.5, 6.5, "#d6eaf8"),
        ("Module 1:\nFeature Engineering", 2.6, 6.5, "#d5f5e3"),
        ("Engineered\nFeature Matrix", 4.7, 6.5, "#fcf3cf"),
        ("Module 2:\nImbalance Baselines", 6.8, 6.5, "#fadbd8"),
        ("Module 3:\nCost-Sensitive\nLightGBM/XGBoost", 8.9, 6.5, "#e8daef"),
        ("Module 4:\nUnsupervised\nAnomaly Detection", 11.0, 6.5, "#d1f2eb"),
        ("Module 5:\nSHAP Explainability", 13.1, 6.5, "#fdebd0"),
    ]

    width = 1.75
    height = 1.0
    for label, x, y, color in pipeline:
        add_box(ax, x, y, width, height, label, color)

    for idx in range(len(pipeline) - 1):
        _, x1, y1, _ = pipeline[idx]
        _, x2, y2, _ = pipeline[idx + 1]
        add_arrow(ax, (x1 + width, y1 + height / 2), (x2, y2 + height / 2))

    dashboard_x = 6.2
    dashboard_y = 3.2
    dashboard_width = 3.6
    dashboard_height = 1.15
    add_box(
        ax,
        dashboard_x,
        dashboard_y,
        dashboard_width,
        dashboard_height,
        "Streamlit Fraud Analyst Dashboard",
        "#d6dbdf",
    )
    add_arrow(ax, (13.95, 6.5), (dashboard_x + dashboard_width, dashboard_y + dashboard_height))

    dashboard_views = [
        ("Portfolio\nOverview", 4.2, 1.6),
        ("Single Transaction\nExplainer", 6.8, 1.6),
        ("Project\nMethodology", 9.4, 1.6),
    ]
    for label, x, y in dashboard_views:
        add_box(ax, x, y, 2.1, 0.9, label, "#ebedef")
        add_arrow(ax, (dashboard_x + dashboard_width / 2, dashboard_y), (x + 1.05, y + 0.9))

    side_outputs = [
        ("Figures saved to\noutputs/figures/", 3.5, 4.7, "#f4f6f7"),
        ("Reports saved to\noutputs/reports/", 6.4, 4.7, "#f4f6f7"),
        ("Best model saved to\nmodels/saved/", 9.3, 4.7, "#f4f6f7"),
    ]
    for label, x, y, color in side_outputs:
        add_box(ax, x, y, 2.35, 0.85, label, color)

    add_arrow(ax, (7.65, 6.5), (4.65, 5.55), color="#7f8c8d")
    add_arrow(ax, (9.75, 6.5), (7.58, 5.55), color="#7f8c8d")
    add_arrow(ax, (10.65, 6.5), (10.48, 5.55), color="#7f8c8d")

    ax.text(
        8,
        0.55,
        "Pipeline artifacts power an analyst-facing dashboard without retraining models in the app.",
        ha="center",
        va="center",
        fontsize=11,
        color="#566573",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Architecture diagram saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
