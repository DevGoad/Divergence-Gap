"""Write result tables, figures and a Markdown summary."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .models import CONTROLS  # noqa: E402

# Reference palette (dataviz skill): diverging blue <-> red with a gray midpoint;
# ordinal blue ramp for quintiles; recessive ink for axes and text.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
POLE_CORP, POLE_MEDIA, MID = "#e34948", "#2a78d6", "#8a8985"
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True,
})


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def fig_coefficients(reg: pd.DataFrame, out: Path) -> None:
    d = reg[(reg["spec"] == "S2_abs") & reg["var"].isin(["D", "absD"])].copy()
    if d.empty:
        return
    d["family"] = d["y"].str.replace(r"_\d+$", "", regex=True)
    fams = list(dict.fromkeys(d["family"]))
    fig, axes = plt.subplots(1, len(fams), figsize=(3.1 * len(fams), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        for k, (var, color) in enumerate((("D", POLE_CORP), ("absD", INK_2))):
            g = d[(d["family"] == fam) & (d["var"] == var)].sort_values("h")
            x = np.arange(len(g)) + (k - 0.5) * 0.22
            ax.errorbar(x, g["coef"], yerr=1.96 * g["se_dk"], fmt="o", color=color, ms=6,
                        elinewidth=2, capsize=0, label=var)
            ax.set_xticks(np.arange(len(g)), [f"h={h}" for h in g["h"]])
        ax.axhline(0, color=INK_2, lw=1)
        ax.set_title(fam.replace("fwd_", "").replace("_", " "))
    axes[0].set_ylabel("coef per 1 SD (95% CI, Driscoll-Kraay)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["D (signed)", "|D|"], loc="upper right", ncol=2)
    fig.suptitle("Divergence coefficients (spec S2: D, |D|, media tone + controls; firm + week FE)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_sorts(sorts: pd.DataFrame, h: int, out: Path) -> None:
    d = sorts[(sorts["sort"] == "D") & (sorts["h"] == h)].set_index("target")
    tg = [t for t in (f"fwd_log_idio_{h}", f"fwd_dd_{h}", f"anom_dd_{h}", "dvol") if t in d.index]
    if not tg:
        return
    fig, axes = plt.subplots(1, len(tg), figsize=(3.1 * len(tg), 3.2))
    for ax, t in zip(np.atleast_1d(axes), tg):
        vals = [d.loc[t, f"Q{q}"] for q in range(1, 6)]
        ax.bar(range(1, 6), vals, color=ORDINAL, width=0.72, edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(range(1, 6), ["Q1\nmedia\nrosier", "Q2", "Q3", "Q4", "Q5\ncorp\nrosier"], fontsize=8)
        name = t.replace(f"_{h}", "").replace("fwd_", "")
        ax.set_title(f"{name}\n"
                     f"Q5-Q1 = {d.loc[t, 'Q5-Q1']:.3f} (t = {d.loc[t, 't_NW']:.2f})", fontsize=10)
    fig.suptitle(f"Forward risk by divergence quintile (h={h} days)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_event(ev: pd.DataFrame, out: Path) -> None:
    if ev.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    styles = {"corp_rosier": (POLE_CORP, "Company rosier than media (top decile D)"),
              "middle": (MID, "Middle"),
              "media_rosier": (POLE_MEDIA, "Media rosier than company (bottom decile D)")}
    for grp, (color, label) in styles.items():
        g = ev[ev["group"] == grp]
        if g.empty:
            continue
        n = int(g["n_events"].iloc[0])
        axes[0].plot(g["day"], 100 * g["car"], color=color, label=f"{label} (n={n})")
        axes[0].fill_between(g["day"], 100 * (g["car"] - 1.96 * g["car_se"]),
                             100 * (g["car"] + 1.96 * g["car_se"]), color=color, alpha=0.12, lw=0)
        axes[1].plot(g["day"], 100 * g["abs_abret"].rolling(5, min_periods=1).mean(), color=color)
    axes[0].axhline(0, color=INK_2, lw=1)
    axes[0].set_title("Cumulative abnormal return, %")
    axes[1].set_title("Mean |abnormal return|, % (5-day smoothed)")
    for ax in axes:
        ax.set_xlabel("trading days after signal")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_timeseries(panel: pd.DataFrame, out: Path) -> None:
    d = panel.dropna(subset=["C", "N"]).groupby("date")[["C", "N"]].mean().rolling(4).mean()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(d.index, d["C"], color=POLE_CORP, label="Corporate tone (type-adjusted)")
    ax.plot(d.index, d["N"], color=POLE_MEDIA, label="Media tone")
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_title("Average FinBERT tone across firms (4-week mean)", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_(no results)_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows]) + "\n"


def write_report(res: dict[str, pd.DataFrame], panel: pd.DataFrame, cfg: dict, out_dir: Path,
                 title: str = "Divergence Gap - results") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    h0 = cfg["analysis"]["primary_horizon"]
    fig_coefficients(res["regressions"], out_dir / "fig_coefficients.png")
    fig_sorts(res["sorts"], h0, out_dir / "fig_sorts.png")
    fig_event(res["event_study"], out_dir / "fig_event_study.png")
    fig_timeseries(panel, out_dir / "fig_tone.png")

    reg = res["regressions"].copy()
    if not reg.empty:
        reg["coef"] = [f"{c:.4f}{_stars(p)}" for c, p in zip(reg["coef"], reg["p_dk"])]
    main = reg[reg["h"] == h0][["y", "spec", "var", "coef", "t_dk", "t_cl2", "nobs", "r2_within"]] if not reg.empty else reg

    d = panel.dropna(subset=["D"])
    lines = [
        f"# {title}", "",
        f"Sample: {panel['date'].min():%Y-%m-%d} to {panel['date'].max():%Y-%m-%d}; "
        f"{d['ticker'].nunique()} firms; {len(d):,} firm-weeks with both streams "
        f"(of {len(panel):,}). Standardization: `{cfg['panel']['standardize']}`.", "",
        "**D** = z(corporate tone) - z(media tone): positive means the company sounds rosier than the press. "
        "Drawdown targets are depths (positive = worse). All regressions include firm and week fixed effects "
        "and controls: " + ", ".join(f"`{c}`" for c in CONTROLS) + ".", "",
        "Significance stars use Driscoll-Kraay p-values: * 10%, ** 5%, *** 1%. `t_cl2` = two-way (firm, week) clustered t.", "",
        f"## 1. Panel regressions (h = {h0} trading days)", "", _md(main),
        "![coefficients](fig_coefficients.png)", "",
        "## 2. Anomalous drawdowns - logit (year dummies, firm-clustered)", "", _md(res["logit"]),
        "## 3. Dumitrescu-Hurlin panel Granger causality (weekly)", "",
        "Rows with `y=log_rv_5, x=D` test divergence -> volatility; reversed rows test volatility -> divergence "
        "(reverse causality).", "", _md(res["granger"]),
        "## 4. Quintile sorts on D", "", _md(res["sorts"][res["sorts"]["sort"] == "D"]),
        "![sorts](fig_sorts.png)", "",
        "## 5. Out-of-sample (expanding window, embargoed)", "",
        "Base = controls + media tone; augmented = base + D + |D|. Clark-West tests whether the augmented "
        "model forecasts better (one-sided).", "", _md(res["oos"]),
        "AUC for anomalous drawdowns by test year:", "", _md(res["oos_auc"]),
        "## 6. Placebo: D permuted across firms within each week", "", _md(res["permutation"]),
        "## 7. Event study", "", "![event study](fig_event_study.png)", "",
        "![tone](fig_tone.png)", "",
    ]
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
