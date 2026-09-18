"""Figures 1-3 from the analysis results.

Figure 1: study flow diagram (results/01_descriptive.xlsx, sheet flow).
Figure 2: adjusted odds ratio for 30-day mortality across the correction rate, restricted cubic
          spline, reference 6 mmol/L/24 h (results/05_sensitivity.xlsx, sheet spline_dose_response).
Figure 3: age-specific standardized 30-day mortality (A) and 14-day cumulative incidence of
          neurological deterioration (B) by correction-rate category, from the interaction models
          (results/02_primary_model.xlsx, sheet figure3_age_grid; results/03_neurological.xlsx,
          sheet figure3_age_grid_interaction).

Output: results/figures/Fig1_flow_diagram, Fig2_correction_rate_spline, Fig3_age_panels (.png, .pdf, .tif)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyponatremia.config import RATE_LEVELS, RESULTS

OUT = os.path.join(RESULTS, "figures")
os.makedirs(OUT, exist_ok=True)
LABEL = {"<4": "<4", "4-8": "4–8", ">8-10": ">8 to 10", ">10": ">10"}
COLOR = {"<4": "#1f77b4", "4-8": "#2ca02c", ">8-10": "#ff7f0e", ">10": "#d62728"}
FORMATS = ["png", "pdf", "tif"]


def save(fig, name, tight=False):
    for ext in FORMATS:
        path = os.path.join(OUT, f"{name}.{ext}")
        kwargs = {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {"dpi": 300}
        if tight:
            kwargs["bbox_inches"] = "tight"
        fig.savefig(path, **kwargs)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def flow_diagram(flow):
    a = dict(zip(flow["Step"], flow["n"]))
    excluded = ["Excluded: sodium <125 mmol/L not confirmed", "Excluded: laboratory artefact", "Excluded: glucose-corrected sodium >=125 mmol/L"]
    n_excluded = sum(a[k] for k in excluded)
    fig, ax = plt.subplots(figsize=(6.9, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10.5)
    ax.set_ylim(12.9, 22)

    def box(y, text, x=1.0, w=5.4, h=1.15, bold=False):
        ax.add_patch(plt.Rectangle((x, y - h), w, h, fill=False, lw=1.2))
        ax.text(x + w / 2, y - h / 2, text, ha="center", va="center", fontsize=8.5, weight="bold" if bold else "normal")

    def side(y, text, x=6.5, w=3.95, h=1.15):
        ax.add_patch(plt.Rectangle((x, y - h), w, h, fill=False, lw=0.9, ls="--"))
        ax.text(x + w / 2, y - h / 2, text, ha="center", va="center", fontsize=6.8)
        ax.annotate("", xy=(x, y - h / 2), xytext=(6.4, y - h / 2), arrowprops=dict(arrowstyle="->", lw=0.9))

    def arrow(y1, y2):
        ax.annotate("", xy=(3.7, y2), xytext=(3.7, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

    y = 21.5
    box(y, "Adult emergency department visits with\nserum sodium <125 mmol/L, 2018–2024\nn = %d" % a["Adult ED visits with serum sodium <125 mmol/L"])
    side(y, "Excluded, n = %d\nSodium <125 mmol/L not confirmed: %d\nLaboratory artefact: %d\nGlucose-corrected sodium ≥125: %d"
         % (n_excluded, a[excluded[0]], a[excluded[1]], a[excluded[2]]), h=1.5)
    arrow(y - 1.15, y - 1.6)
    y -= 1.6
    box(y, "Eligible visits\nn = %d" % a["Eligible visits"])
    side(y, "No 24-hour sodium measurement\nn = %d" % a["No 24-hour sodium measurement"])
    arrow(y - 1.15, y - 1.6)
    y -= 1.6
    box(y, "Visits with a calculable correction rate\nn = %d" % a["Visits with a calculable correction rate"])
    side(y, "Later eligible visits of the same patient\nn = %d" % (a["Visits with a calculable correction rate"] - a["First eligible visit per patient"]))
    arrow(y - 1.15, y - 1.6)
    y -= 1.6
    box(y, "First eligible visit per patient\nn = %d" % a["First eligible visit per patient"])
    side(y, "Death before 24 h\n(exposure window incomplete)\nn = %d" % a["Death before the landmark"])
    arrow(y - 1.15, y - 1.6)
    y -= 1.6
    box(y, "Study population\n(24-hour landmark cohort)\nn = %d" % a["Study population (landmark cohort)"], h=1.4, bold=True)
    return fig


# ---------------------------------------------------------------- Figure 2
def spline_figure(spline):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = spline["rate_mmol_L_24h"]
    ax.plot(x, spline["OR_vs_6"], color="#1f77b4", lw=2)
    ax.fill_between(x, spline["lower"], spline["upper"], color="#1f77b4", alpha=0.15)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Sodium correction rate (mmol/L per 24 h)")
    ax.set_ylabel("Adjusted odds ratio for 30-day mortality\n(reference 6 mmol/L per 24 h)")
    for v in [4, 8, 10]:
        ax.axvline(v, color="grey", lw=0.6, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Figure 3
def age_panel(ax, G, ylabel, title, ymax):
    for lvl in RATE_LEVELS:
        ax.plot(G["age"], G[lvl], color=COLOR[lvl], lw=2, label=LABEL[lvl])
        ax.fill_between(G["age"], G[f"{lvl} lower"], G[f"{lvl} upper"], color=COLOR[lvl], alpha=0.12)
    ax.set_xlabel("Age (years)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(30, 95)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for x in [65, 75, 85]:
        ax.axvline(x, color="grey", lw=0.6, ls=":")


def age_panels(GA, GB):
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 7.6))
    age_panel(axes[0], GA, "Adjusted 30-day mortality risk (%)", "A", 60)
    age_panel(axes[1], GB, "Adjusted 14-day cumulative incidence of\nneurological deterioration (%)", "B", 45)
    axes[0].legend(title="Correction rate (mmol/L per 24 h)", frameon=False, fontsize=8, title_fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    flow = pd.read_excel(os.path.join(RESULTS, "01_descriptive.xlsx"), sheet_name="flow")
    spline = pd.read_excel(os.path.join(RESULTS, "05_sensitivity.xlsx"), sheet_name="spline_dose_response")
    GA = pd.read_excel(os.path.join(RESULTS, "02_primary_model.xlsx"), sheet_name="figure3_age_grid")
    GB = pd.read_excel(os.path.join(RESULTS, "03_neurological.xlsx"), sheet_name="figure3_age_grid_interaction")
    save(flow_diagram(flow), "Fig1_flow_diagram", tight=True)
    save(spline_figure(spline), "Fig2_correction_rate_spline")
    save(age_panels(GA, GB), "Fig3_age_panels")
    print("figures written to", OUT)
