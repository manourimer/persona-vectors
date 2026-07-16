"""
Generate paper PDF: Persona Vectors as a Psychometric Instrument.
Outputs: outputs/persona_vectors_paper.pdf
All findings reflect corrected real Gemma-3-12B activations (June 29 re-extraction).
"""

from __future__ import annotations
import io, sys, tempfile, os, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "outputs" / "persona_vectors_paper.pdf"

# ── colours ──────────────────────────────────────────────────────────────────
BLUE  = colors.HexColor("#2563EB")
DBLUE = colors.HexColor("#1E3A5F")
LBLUE = colors.HexColor("#DBEAFE")
GREY  = colors.HexColor("#374151")
LGREY = colors.HexColor("#F3F4F6")
AMBER = colors.HexColor("#D97706")
GREEN = colors.HexColor("#059669")
WHITE = colors.white

TRAIT_COLORS = {
    "honesty":       "#2563EB",
    "harmlessness":  "#059669",
    "fairness":      "#D97706",
    "compassion":    "#9333EA",
}

# ── styles ────────────────────────────────────────────────────────────────────
def S(name, **kw): return ParagraphStyle(name, **kw)

Title    = S("T",  fontSize=19, textColor=DBLUE, alignment=TA_CENTER, leading=25, fontName="Helvetica-Bold", spaceAfter=6)
Authors  = S("Au", fontSize=11, textColor=GREY,  alignment=TA_CENTER, leading=16, fontName="Helvetica")
Affil    = S("Af", fontSize=9,  textColor=GREY,  alignment=TA_CENTER, leading=12, fontName="Helvetica-Oblique")
SectHead = S("SH", fontSize=13, textColor=DBLUE, leading=18, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)
SubHead  = S("SB", fontSize=10.5, textColor=DBLUE, leading=14, fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
Body     = S("Bo", fontSize=9.5, textColor=GREY, leading=14, fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=6)
Bullet   = S("Bu", fontSize=9.5, textColor=GREY, leading=13, fontName="Helvetica", leftIndent=14, spaceAfter=3, bulletIndent=6)
Caption  = S("Ca", fontSize=8.5, textColor=GREY, leading=12, fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=6)
Abstract = S("Ab", fontSize=9.5, textColor=GREY, leading=13, fontName="Helvetica", alignment=TA_JUSTIFY)
Note     = S("No", fontSize=8.5, textColor=AMBER, leading=12, fontName="Helvetica-Oblique", alignment=TA_JUSTIFY)

W, H = LETTER
MARGIN = 0.85 * inch

# ── page callbacks ────────────────────────────────────────────────────────────
def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BLUE); canvas.setLineWidth(1.5)
    canvas.line(MARGIN, H-0.55*inch, W-MARGIN, H-0.55*inch)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(GREY)
    canvas.drawString(MARGIN, H-0.45*inch, "Persona Vectors as a Psychometric Instrument for Moral Character in LLMs")
    canvas.drawRightString(W-MARGIN, H-0.45*inch, "Manou Rimer · BlueDot Impact · 2026")
    canvas.setLineWidth(0.5); canvas.setStrokeColor(LGREY)
    canvas.line(MARGIN, 0.48*inch, W-MARGIN, 0.48*inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(W/2, 0.3*inch, f"{doc.page}")
    canvas.restoreState()

def first_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DBLUE); canvas.rect(0, H-0.35*inch, W, 0.35*inch, stroke=0, fill=1)
    canvas.setFillColor(BLUE);  canvas.rect(0, H-0.42*inch, W, 0.07*inch, stroke=0, fill=1)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(WHITE)
    canvas.drawRightString(W-MARGIN, H-0.22*inch, "BlueDot Impact Technical AI Safety Project Sprint · 2026")
    canvas.setFillColor(GREY); canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(W/2, 0.3*inch, f"{doc.page}")
    canvas.restoreState()

# ── helpers ───────────────────────────────────────────────────────────────────
_tmp_pngs = []

def fig_img(fig, width=5.5*inch):
    from PIL import Image as PILImage
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    _tmp_pngs.append(tmp.name)
    fig.savefig(tmp.name, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pw, ph = PILImage.open(tmp.name).size
    height = width * ph / pw
    img = Image(tmp.name, width=width, height=height)
    img.hAlign = "CENTER"
    return img

def stbl(data, col_widths=None, hbg=DBLUE):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), hbg),
        ("TEXTCOLOR",  (0,0),(-1,0), WHITE),
        ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),(-1,-1), 8),
        ("LEADING",    (0,0),(-1,-1), 11),
        ("TOPPADDING", (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",(0,0),(-1,-1), 6),
        ("RIGHTPADDING",(0,0),(-1,-1), 6),
        ("GRID",       (0,0),(-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGREY]),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
    ]))
    return t

def amber_box(text):
    t = Table([[Paragraph(text, Note)]], colWidths=[W-2*MARGIN-0.4*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#FEF3C7")),
        ("BOX",(0,0),(-1,-1), 1.2, AMBER),
        ("TOPPADDING",(0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING",(0,0),(-1,-1), 10),
        ("RIGHTPADDING",(0,0),(-1,-1), 10),
    ]))
    return t

# ── load data ─────────────────────────────────────────────────────────────────
struct   = pd.read_csv(ROOT/"outputs/structure_analysis/structure_summary.csv")
corr32   = pd.read_csv(ROOT/"outputs/structure_analysis/correlation_matrix_layer32.csv", index_col=0)
corr40   = pd.read_csv(ROOT/"outputs/structure_analysis/correlation_matrix_layer40.csv", index_col=0)
rel_sum  = pd.read_csv(ROOT/"outputs/reliability_analysis/reliability_summary.csv")
d_study  = pd.read_csv(ROOT/"outputs/reliability_analysis/d_study_results.csv")
contrast = pd.read_csv(ROOT/"outputs/controls/contrast_validation_positive_control.csv")
rand_vec = pd.read_csv(ROOT/"outputs/controls/random_vector_control_summary.csv")
shuf     = pd.read_csv(ROOT/"outputs/controls/shuffled_label_control_summary.csv")
dup      = pd.read_csv(ROOT/"outputs/controls/exact_duplicate_control_summary.csv")
pre_re   = pd.read_csv(ROOT/"outputs/controls/preprocessing_metric_recompute.csv")
syn_sim  = pd.read_csv(ROOT/"outputs/controls/synonym_vectors/synonym_cosine_similarity.csv")
syn_agr  = pd.read_csv(ROOT/"outputs/controls/synonym_vectors/synonym_projection_agreement.csv")
vec_meta = pd.read_csv(ROOT/"outputs/vector_construction/persona_vector_metadata.csv")
pc32     = pd.read_csv(ROOT/"outputs/structure_analysis/pca_scores_layer32.csv")
pc40     = pd.read_csv(ROOT/"outputs/structure_analysis/pca_scores_layer40.csv")
mvp      = pd.read_parquet(ROOT/"data/processed/ethics_curated_mvp.parquet")

# Verified/recomputed numbers (see scripts/verify_paper_numbers.py) — replaces
# hardcoded literals for AUCs that previously had no traceable source file.
import json as _json
with open(ROOT/"outputs/paper_verification/verified_auc_results.json") as _f:
    VERIFIED = _json.load(_f)
cos4vec  = pd.read_csv(ROOT/"outputs/paper_verification/cosine_similarity_4vec_layer32.csv", index_col=0)
cos8vec  = pd.read_csv(ROOT/"outputs/paper_verification/cosine_similarity_8vec_layer32.csv", index_col=0)

def ci_str(d, decimals=3):
    """Format a {auc, ci_low, ci_high} dict as 'AUC [low, high]'."""
    f = f"{{:.{decimals}f}}"
    return f"{f.format(d['auc'])} [{f.format(d['ci_low'])}, {f.format(d['ci_high'])}]"

def _stratified_diagonal_dominance():
    """Diagonal dominance (annotated trait = top-projecting vector) within each
    ETHICS source split (justice/deontology/commonsense — not the 4-way item-format
    grouping used elsewhere; see paper §2.3). Two-sided z-test vs. chance=0.25."""
    ethics = pd.read_parquet(ROOT/"outputs/ethics_projection/ethics_trait_projections_centered_wide.parquet")
    proj_cols = ["projection_honesty","projection_harmlessness","projection_fairness","projection_compassion"]
    top = ethics[proj_cols].idxmax(axis=1).str.replace("projection_", "", regex=False)
    match = (top == ethics["primary_trait"])
    rows = []
    for fmt, idx in ethics.groupby("source_split").groups.items():
        sub = match.loc[idx]
        n = len(sub)
        dd = sub.mean()
        se = (0.25 * 0.75 / n) ** 0.5
        z = (dd - 0.25) / se
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        rows.append((fmt, n, dd, p))
    return sorted(rows, key=lambda r: -r[2])

DIAG_BY_FORMAT = _stratified_diagonal_dominance()

# ── figures ───────────────────────────────────────────────────────────────────

def fig_eff_dim():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    layers = [32, 40, 47]
    eds = [struct[struct.layer==l]["effective_dimensionality"].values[0] for l in layers]
    bar_colors = ["#2563EB", "#059669", "#2563EB"]
    bars = ax.bar([str(l) for l in layers], eds, color=bar_colors, width=0.4, zorder=3)
    ax.axhline(4.0, color="#9CA3AF", linestyle="--", lw=1, label="Maximum (4.0)")
    ax.set_ylim(0, 4.8); ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Effective Dimensionality", fontsize=9)
    ax.set_title("Participation-Ratio Effective Dimensionality\n(4 traits, 204 ETHICS items)", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4, zorder=0); ax.set_axisbelow(True)
    for i,(l,v) in enumerate(zip(layers,eds)):
        ax.text(i, v+0.07, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1E3A5F")
    ax.annotate("More separation\nat layer 40", xy=(1, eds[1]), xytext=(1.5, 3.2),
                fontsize=8, color="#059669",
                arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2))
    fig.tight_layout()
    return fig

def fig_corr_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    labels = ["Honesty","Harmless.","Fairness","Compassion"]
    for ax, corr, title in zip(axes, [corr32, corr40], ["Layer 32 (ED=1.13)","Layer 40 (ED=2.46)"]):
        mat = corr.values.astype(float)
        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(mat[i,j])>0.7 else "black")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Inter-Trait Projection Correlations", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig

def fig_dstudy():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ks = [1,2,3,4,5]
    for t, c in TRAIT_COLORS.items():
        sub = d_study[(d_study.layer==32) & (d_study.projected_trait==t)]
        gs  = [sub[sub.n_paraphrases==k]["g_coefficient"].values[0] for k in ks]
        ax.plot(ks, gs, marker="o", color=c, label=t.capitalize(), lw=2, ms=5)
    ax.axhline(0.70, color="#6B7280", linestyle="--", lw=1, alpha=0.7, label="G=0.70")
    ax.axhline(0.80, color="#374151", linestyle=":", lw=1, alpha=0.7, label="G=0.80")
    ax.set_xlim(0.8,5.2); ax.set_ylim(0.4,1.0)
    ax.set_xlabel("Number of Paraphrases (k)", fontsize=9)
    ax.set_ylabel("G-Coefficient G(k)", fontsize=9)
    ax.set_title("D-Study: G-Coefficients vs Paraphrases\n(Layer 32 — shared moral-valence axis)", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig

def fig_contrast():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    layers = [16,24,28,32,40,47]; x = np.arange(len(layers)); w = 0.2
    offsets = np.linspace(-1.5*w, 1.5*w, 4)
    for ti, (t, off) in enumerate(zip(TRAIT_COLORS, offsets)):
        sub  = contrast[contrast.trait==t]
        aucs = [sub[sub.layer==l]["auc"].values[0] for l in layers]
        ax.bar(x+off, aucs, width=w, color=TRAIT_COLORS[t], label=t.capitalize(), alpha=0.85)
    ax.axhline(0.75, color="#1F2937", linestyle="--", lw=1.2, label="AUC=0.75")
    ax.set_ylim(0.4,1.08); ax.set_xticks(x); ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Layer", fontsize=9); ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("Contrast Validation AUC\n(separating high vs. low persona construction items)", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig

def fig_pc1_label():
    """PC1 scores split by ethical label and format."""
    ethics = pd.read_parquet(ROOT/"outputs/ethics_projection/ethics_trait_projections_centered_wide.parquet")
    merged = pc32.merge(mvp[["item_id","label"]], on="item_id")
    merged = merged.merge(ethics[["item_id","scenario_text"]], on="item_id")
    merged["label"] = merged["label"].astype(float)

    def fmt(row):
        t, i = str(row["scenario_text"]), str(row["item_id"])
        if "[EXCUSE]" in t: return "EXCUSE"
        if t.strip().startswith(("AITA","WIBTA")): return "AITA"
        if "commonsense" in i: return "Commonsense"
        if "justice" in i: return "Justice"
        return "Other"

    merged["format"] = merged.apply(fmt, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))

    # Left: PC1 distribution by label
    ax = axes[0]
    w_scores = merged[merged.label==1]["PC1"].values
    o_scores = merged[merged.label==0]["PC1"].values
    ax.hist(o_scores, bins=20, alpha=0.6, color="#059669", label="Morally OK (n=98)")
    ax.hist(w_scores, bins=20, alpha=0.6, color="#DC2626", label="Morally Wrong (n=106)")
    ax.set_xlabel("PC1 Score", fontsize=9); ax.set_ylabel("Count", fontsize=9)
    _p = VERIFIED["pooled"]
    ax.set_title(f"PC1 by Ethical Label\n(overall AUC={_p['auc']:.3f} "
                 f"[{_p['ci_low']:.2f}, {_p['ci_high']:.2f}], r={_p['pearson_r_with_label']:.3f})",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    # Right: AUC by format, with bootstrap 95% CI error bars
    ax2 = axes[1]
    formats = ["Justice","Commonsense","EXCUSE","AITA"]
    fmt_results = [VERIFIED["by_format"][f] for f in formats]
    aucs_fmt = [r["auc"] for r in fmt_results]
    err_lo = [r["auc"] - r["ci_low"] for r in fmt_results]
    err_hi = [r["ci_high"] - r["auc"] for r in fmt_results]
    bar_c = ["#2563EB","#059669","#D97706","#9333EA"]
    bars = ax2.barh(formats, aucs_fmt, color=bar_c, alpha=0.85,
                     xerr=[err_lo, err_hi], capsize=3, error_kw=dict(ecolor="#1F2937", elinewidth=1.1))
    ax2.axvline(0.5, color="#6B7280", linestyle="--", lw=1, label="Chance (0.50)")
    ax2.set_xlim(0.1, 0.95)
    ax2.set_xlabel("AUC (PC1 predicting morally wrong), 95% CI", fontsize=9)
    ax2.set_title("AUC by Item Format\n(bootstrap 95% CI, n_boot=5000)", fontsize=9.5, fontweight="bold")
    for bar, v in zip(bars, aucs_fmt):
        ax2.text(v+0.01, bar.get_y()+bar.get_height()/2, f"{v:.3f}",
                 va="bottom", ha="left", fontsize=8, fontweight="bold")
    ax2.legend(fontsize=8); ax2.xaxis.grid(True, alpha=0.3); ax2.set_axisbelow(True)

    fig.tight_layout()
    return fig

def fig_8vec_corr():
    """8x8 correlation matrix of all original + synonym projections."""
    orig = pd.read_parquet(ROOT/"outputs/ethics_projection/ethics_trait_projections_centered_wide.parquet")
    syn  = pd.read_csv(ROOT/"outputs/controls/synonym_vectors/synonym_ethics_projections_layer32.csv")
    merged = orig.merge(syn, on="item_id")
    cols   = ["projection_honesty","projection_harmlessness","projection_fairness","projection_compassion",
              "projection_truthfulness","projection_harm_avoidance","projection_impartiality","projection_empathy"]
    labels = ["Honesty","Harmless.","Fairness","Compassion","Truthful.","HarmAvoid","Impartial","Empathy"]
    mat = merged[cols].corr().values

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(mat[i,j])>0.7 else "black")
    ax.set_title("8-Vector Projection Correlation Matrix\n(4 original + 4 synonym, layer 32, ED=1.19)",
                 fontsize=9.5, fontweight="bold")
    ax.axvline(3.5, color="white", lw=1.5); ax.axhline(3.5, color="white", lw=1.5)
    fig.tight_layout()
    return fig

def fig_cosine_matrix():
    """4x4 cosine similarity matrix of the original persona vectors themselves
    (activation space, layer 32) — vector geometry, not projection scores."""
    labels = ["Honesty","Harmless.","Fairness","Compassion"]
    mat = cos4vec.values.astype(float)

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(mat[i,j])>0.7 else "black")
    ax.set_title("Vector Cosine Similarity\n(activation space, layer 32)", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    return fig

# ── build ─────────────────────────────────────────────────────────────────────
def build():
    doc = BaseDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.0*inch, bottomMargin=0.7*inch,
    )
    frame  = Frame(MARGIN, 0.7*inch, W-2*MARGIN, H-1.7*inch, id="normal")
    cframe = Frame(MARGIN, 0.6*inch, W-2*MARGIN, H-1.1*inch, id="cover")
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cframe], onPage=first_page),
        PageTemplate(id="Body",  frames=[frame],  onPage=header_footer),
    ])

    story = []

    # ── COVER ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "Persona Vectors as a Psychometric Instrument<br/>for Moral Character in Language Models",
        Title))
    story.append(Spacer(1, 0.08*inch))
    story.append(Paragraph("Manou Rimer", Authors))
    story.append(Paragraph("BlueDot Impact Technical AI Safety Project Sprint", Affil))
    story.append(Paragraph("2026", Affil))
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="85%", thickness=1.5, color=BLUE, hAlign="CENTER"))
    story.append(Spacer(1, 0.16*inch))

    abstract_text = (
        "We apply the persona vector method of Chen et al. (2025) to four moral character traits — "
        "<b>honesty</b>, <b>harmlessness</b>, <b>fairness</b>, and <b>compassion</b> — in "
        "Gemma-3-12B-IT, and conduct, to our knowledge, the first systematic psychometric "
        "evaluation of the resulting measurement structure. Using a contrastive system-prompt "
        "paradigm, we construct difference-of-means persona vectors at six transformer layers and "
        "project 204 curated ETHICS benchmark items onto these vectors. "
        "Contrary to our initial hypothesis, structural analysis reveals that the four trait "
        "projections collapse onto a single dominant measurement dimension at layers 32 and 47 "
        "(effective dimensionality ≈ 1.13, PC1 explaining 94% of variance, mean inter-trait "
        "projection |r| = 0.92), with only modest separation at layer 40 (ED = 2.46). "
        "This shared-dimension pattern extends to synonym trait vectors: all 8 vectors "
        "(4 original + 4 synonym) produced an 8×8 projection-score correlation matrix with "
        "ED = 1.19, indicating the projections converge onto one dimension regardless of trait "
        "label — though the persona vectors themselves are only moderately cosine-similar in "
        "activation space (see §3.2), so this is best read as a measurement-structure finding "
        "rather than a claim that the underlying vectors are geometrically collinear. "
        "Generalizability theory analysis on 761 paraphrase variants yields G(k=3) ≥ 0.74 — "
        "demonstrating that the method reliably measures <i>something</i>, but that something "
        "may reflect a shared moral-salience axis rather than four trait-specific directions "
        "(a tentative interpretation; see §4.3). "
        "This shared axis predicts ground-truth ethical labels at AUC = 0.585 (95% CI [0.51, 0.66]), "
        "with substantial variation by item format (justice: AUC = 0.71, 95% CI [0.57, 0.84]; "
        "AITA: AUC = 0.38, 95% CI [0.19, 0.58]) — a format-stratified analysis shows this pooled "
        "figure conceals a signal concentrated in justice-format items specifically, though several "
        "format-level intervals are wide given small per-format samples. "
        "To test whether the collapse is an artefact of ETHICS' item-format confound, we built an "
        "independent, format-controlled item bank (160 items, single format, matched upheld/violated "
        "pairs, zero trait-name leakage) that removes this specific confound: the same collapse "
        "replicates almost exactly, including the same most-correlated trait pair at the same layers. "
        "We then directly tested one candidate explanation for the collapse — that it reflects a "
        "generic RLHF-instilled 'aligned vs. unaligned persona' axis rather than trait-specific "
        "content — by constructing a control vector (virtue_axis) from deliberately generic, "
        "non-trait-specific contrastive prompts. This vector separates its own high/low construction "
        "items more cleanly than any trait vector (AUC up to 1.00 on that internal construction-set "
        "check, not independent criterion validity), yet is nearly orthogonal to the actual shared "
        "collapse direction (cosine ≈ 0 to −0.31 across layers) and, at the two layers where the "
        "collapse is strongest, its addition <i>increases</i> effective dimensionality rather than "
        "leaving it unchanged — evidence against the specific hypothesis that this particular "
        "virtue-axis direction explains the shared structure, though it does not rule out other "
        "generic or multidimensional alignment representations. "
        "Our findings formally quantify and extend the cross-trait correlation pattern noted in "
        "a footnote by Chen et al., show that the collapse is not an artefact of the ETHICS "
        "format–trait confound specifically, and show that this one natural candidate explanation "
        "does not survive direct empirical test — leaving the precise cause of the collapse an open "
        "question for AI psychometrics."
    )
    abox = Table([[Paragraph(abstract_text, Abstract)]], colWidths=[W-2*MARGIN-0.4*inch])
    abox.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), LBLUE),
        ("TOPPADDING",(0,0),(-1,-1), 10), ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING",(0,0),(-1,-1), 14), ("RIGHTPADDING",(0,0),(-1,-1), 14),
        ("BOX",(0,0),(-1,-1), 1.5, BLUE),
    ]))
    story.append(abox)
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "<b>Keywords:</b> mechanistic interpretability, persona vectors, moral character, "
        "psychometrics, generalizability theory, representation engineering, LLM alignment",
        S("KW", parent=Body, fontSize=8.5, spaceAfter=4)))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LGREY, hAlign="CENTER"))

    # ── §1 INTRODUCTION ──────────────────────────────────────────────────────
    story.append(NextPageTemplate("Body")); story.append(PageBreak())
    story.append(Paragraph("1. Introduction and Motivation", SectHead))
    story.append(Paragraph(
        "Chen et al. (2025) introduced an automated pipeline for extracting <i>persona vectors</i> — "
        "linear directions in a language model's residual stream that correspond to personality "
        "traits such as evil, sycophancy, and propensity to hallucinate. Their method, based on "
        "contrastive difference-of-means activations, enables both monitoring and steering of "
        "persona-related behaviour during deployment and finetuning. In a footnote, they observe "
        "that persona shifts across their traits tend to correlate highly, and suspect this is "
        "partly due to correlations between the underlying persona vectors — but they do not "
        "investigate this systematically.", Body))
    story.append(Paragraph(
        "This work applies the Chen et al. pipeline to four moral character traits central to AI "
        "alignment — <b>honesty</b>, <b>harmlessness</b>, <b>fairness</b>, and <b>compassion</b> — "
        "and conducts, to our knowledge, the first systematic psychometric evaluation of the "
        "resulting measurement structure (a claim scoped to the literature known to us at the "
        "time of writing; see the Unresolved Issues note on a final search). Our goal is to "
        "assess whether persona vectors can function as a reliable, "
        "trait-specific psychometric instrument for moral character, or whether they collapse "
        "onto a shared representation.", Body))

    story.append(Paragraph("1.1 Research Questions", SubHead))
    for q in [
        "<b>RQ1 (Structure):</b> Do the four trait projections form a near-independent "
        "four-dimensional structure, or collapse onto a shared moral axis?",
        "<b>RQ2 (Reliability):</b> Are projection scores reliable across paraphrastic variants "
        "of the same scenario?",
        "<b>RQ3 (Validity):</b> Does the shared axis — if one exists — predict ground-truth "
        "ethical labels, and does this vary by item format?",
    ]:
        story.append(Paragraph(f"• {q}", Bullet))

    # ── §2 METHODOLOGY ───────────────────────────────────────────────────────
    story.append(Paragraph("2. Methodology", SectHead))

    story.append(Paragraph("2.1 Model", SubHead))
    story.append(Paragraph(
        "We use <b>Gemma-3-12B-IT</b> (Gemma Team, Google DeepMind, 2025): 12B parameters, "
        "hidden dimension d = 3,840, 48 transformer blocks (config <i>num_hidden_layers</i> = 48), "
        "indexed 0–47. All forward passes run on NVIDIA A100 GPUs via Modal, using Hugging Face "
        "<i>transformers</i> with <i>output_hidden_states=True</i>, which returns 49 residual-stream "
        "snapshots per forward pass: index 0 is the embedding output (before block 0), and index i "
        "(for i = 1…48) is the residual stream immediately after transformer block i−1. We refer to "
        "a saved activation at \"layer N\" as <i>hidden_states[N]</i> in this indexing — so \"layer 47\" "
        "is the residual stream after block 46 (the second-to-last of 48 blocks), not after the "
        "final block (block 47, which would be <i>hidden_states[48]</i>); \"layer 40\" is "
        "correspondingly after block 39, and so on. "
        "This is a valid and commonly-used probing point (it captures a late, near-final "
        "representation) but we note the off-by-one explicitly here since \"layer 47\" could "
        "otherwise be misread as \"the final block's output,\" which would instead be "
        "<i>hidden_states[48]</i> (not extracted in this study). "
        "Activations are extracted as float32 vectors from this residual-stream hook at the "
        "<i>last prompt token</i> for ETHICS projection and <i>mean over response tokens</i> for "
        "vector construction, following Chen et al.", Body))

    story.append(Paragraph("2.2 Persona Vector Construction (Chen et al. 2025 Pipeline)", SubHead))
    story.append(Paragraph(
        "For each of the four traits, five contrastive system-prompt pairs are generated "
        "(high-persona vs. low-persona). Forty elicitation questions per trait (20 extraction, "
        "20 validation) are posed under each system prompt. Responses are scored by an LLM "
        "annotator (Qwen2.5-7B) and filtered to retain only those confirming intended persona "
        "alignment. The unit-normalised difference of means across retained positive and negative "
        "pole activations defines the persona vector at each layer:", Body))
    story.append(Paragraph(
        "v<sub>trait,layer</sub> = normalise( <greek>m</greek><sub>high</sub> <greek>-</greek> "
        "<greek>m</greek><sub>low</sub> )",
        S("Eq", parent=Body, fontName="Courier", alignment=TA_CENTER, spaceAfter=6)))
    story.append(Paragraph(
        "Vectors are computed at six candidate layers (16, 24, 28, 32, 40, 47). "
        "Prior to projection, the global mean activation across ETHICS items is subtracted "
        "per layer to remove the dominant shared component (~61,000 L<sub>2</sub>-norm).", Body))

    vm = vec_meta[vec_meta.layer.isin([32,40,47])][["trait","layer","n_positive","n_negative","hidden_dim"]]
    td = [["Trait","Layer","Pos. Responses","Neg. Responses","Hidden Dim"]]
    for _, r in vm.iterrows():
        td.append([r.trait.capitalize(), str(int(r.layer)),
                   str(int(r.n_positive)), str(int(r.n_negative)), str(int(r.hidden_dim))])
    story.append(KeepTogether([
        stbl(td, col_widths=[1.1*inch,0.65*inch,1.1*inch,1.1*inch,1.0*inch]),
        Spacer(1,3),
        Paragraph("Table 1. Persona vector construction summary (primary analysis layers).", Caption),
    ]))

    story.append(Paragraph("2.3 ETHICS Benchmark Projection", SubHead))
    story.append(Paragraph(
        "We project 204 curated ETHICS benchmark items (Hendrycks et al., 2021) onto each "
        "persona vector. Items were curated with ground-truth ethical labels "
        "(1 = morally wrong, 0 = morally OK) and primary trait assignments. "
        "This paper uses two distinct groupings of the same 204 items, and we name them "
        "differently on purpose to avoid conflating them. <b>ETHICS source split</b> "
        "(<i>source_split</i> in the data: <i>deontology</i>, <i>justice</i>, <i>commonsense</i>) "
        "is Hendrycks et al.'s own three-way category for how each item was collected (74, 53, "
        "and 77 items respectively). <b>Item format</b> is our own finer four-way text-pattern "
        "classification — <i>EXCUSE</i> (74 items, 96% honesty), <i>justice</i> (53 items, 72% "
        "fairness), <i>commonsense</i> (42 items, 60% harmlessness), and <i>AITA</i> (35 items, "
        "mixed traits) — derived from scenario text and item ID (§ code: <i>fmt()</i> in the paper "
        "build script). The two groupings coincide exactly for <i>deontology</i> = <i>EXCUSE</i> "
        "(the same 74 items under two names — deontology is the ETHICS split name, EXCUSE is our "
        "format label for its distinctive templated phrasing) and for <i>justice</i> = "
        "<i>justice</i>, but <b>not</b> for the third: the <i>commonsense</i> source split (77 "
        "items) splits further into our <i>commonsense</i> <i>format</i> (42 items) and our "
        "<i>AITA</i> format (35 items, first-person Reddit-style posts), which we separate because "
        "they differ qualitatively in structure. Tables and figures using per-format AUC (Table 5) "
        "use the four-way format grouping; Table 7's per-split diagonal dominance uses the "
        "three-way source-split grouping — the sample sizes differ accordingly (e.g. "
        "\"commonsense\" is n=42 in Table 5 but n=77 in Table 7) and this is not an error. "
        "The format–trait confound is a structural property "
        "of the ETHICS benchmark and is reported as a limitation throughout.", Body))

    story.append(Paragraph("2.4 Structural Analysis", SubHead))
    story.append(Paragraph(
        "Structure is assessed via: (i) participation-ratio effective dimensionality "
        "(Σλ)<super>2</super>/Σλ<super>2</super>; (ii) PC1 variance explained; "
        "(iii) mean absolute off-diagonal inter-trait correlation. "
        "Parallel analysis (500 permutations) determines statistically justified component retention.", Body))

    story.append(Paragraph("2.5 Generalizability Theory (G-Theory)", SubHead))
    story.append(Paragraph(
        "To assess reliability across paraphrastic variants, 761 LLM-generated paraphrases "
        "of ETHICS items (Qwen2.5-7B, Stage 4B) were validated for semantic equivalence "
        "and projected onto persona vectors at layers 32/40/47 (Stage 4C). "
        "A one-way random-effects ANOVA decomposes variance into between-item (universe score) "
        "and within-item (paraphrase error) components, yielding G(k) coefficients via D-study.", Body))

    story.append(Paragraph("2.6 Validation Controls", SubHead))
    for ctrl, desc in [
        ("<b>Random vector null</b>", "500 random unit vectors; establishes baseline ED and reliability."),
        ("<b>Shuffled-label specificity</b>", "10,000 label permutations; tests whether real trait assignment is non-random."),
        ("<b>Exact-duplicate ceiling</b>", "Identical prompt twice; verifies G=1.0 upper bound."),
        ("<b>Contrast validation</b>", "AUC separating high vs. low persona construction items (threshold ≥ 0.75)."),
        ("<b>Synonym vector convergent validity</b>", "4 synonym traits via identical pipeline; cosine similarity and projection agreement."),
        ("<b>Preprocessing robustness</b>", "Raw vs. mean-centred activations compared."),
        ("<b>Ground-truth label prediction</b>", "PC1 vs. ethical right/wrong labels; assessed overall and by item format."),
    ]:
        story.append(Paragraph(f"• {ctrl}: {desc}", Bullet))

    # ── §3 FINDINGS ──────────────────────────────────────────────────────────
    story.append(Paragraph("3. Findings", SectHead))

    # 3.1 Structure
    story.append(Paragraph("3.1 Structure: A Single Shared Moral-Valence Axis", SubHead))
    story.append(Paragraph(
        "The four trait projections do <b>not</b> form a near-independent four-dimensional "
        "structure. Instead they collapse onto a single dominant dimension at layers 32 and 47, "
        "with only partial separation at layer 40.", Body))

    story.append(Paragraph(
        "All results below use real Gemma-3-12B-IT activations "
        "(dim=3840); an earlier mock-data run is superseded and discussed in §4.5.",
        S("DI", parent=Caption, alignment=TA_JUSTIFY, textColor=AMBER, fontSize=8)))
    story.append(Spacer(1, 0.06*inch))

    sd = [["Layer","Eff. Dim (max=4)","PC1 Variance","Mean |r|","Max |r|","Most Correlated Pair"]]
    for _, r in struct.iterrows():
        sd.append([str(int(r.layer)), f"{r.effective_dimensionality:.3f}",
                   f"{r.first_pc_variance:.3f}", f"{r.mean_abs_off_diag_corr:.3f}",
                   f"{r.max_abs_trait_corr:.3f}", str(r.most_correlated_pair)])
    story.append(KeepTogether([
        stbl(sd, col_widths=[0.55*inch,1.1*inch,0.95*inch,0.75*inch,0.75*inch,1.65*inch]),
        Spacer(1,3),
        Paragraph("Table 2. Stage 4A structure summary using real Gemma-3-12B activations. "
                  "ED ≈ 1.13 at layers 32 and 47 indicates a single dominant dimension. "
                  "Layer 40 shows more separation (ED = 2.46).", Caption),
    ]))
    story.append(Spacer(1, 0.06*inch))
    story.append(KeepTogether([
        fig_img(fig_eff_dim(), width=4.5*inch),
        Paragraph("Figure 1. Effective dimensionality at layers 32, 40, and 47. "
                  "ED ≈ 1.13 at layers 32/47 indicates near-complete collapse onto one axis. "
                  "Layer 40 (green) shows the most trait separation (ED = 2.46).", Caption),
    ]))
    story.append(Spacer(1, 0.06*inch))
    story.append(KeepTogether([
        fig_img(fig_corr_heatmap(), width=6.0*inch),
        Paragraph("Figure 2. Inter-trait <i>projection-score</i> correlation matrices at layers 32 "
                  "(left) and 40 (right) — Pearson r between how each pair of trait vectors scores "
                  "the same 204 ETHICS items, not a direct measure of the vectors' own geometry "
                  "(see Figure 3b for cosine similarity). "
                  "Layer 32: all absolute projection correlations ≥ 0.87, indicating the four "
                  "trait projections are highly redundant on this item set. "
                  "Layer 40: more modest correlations, mean |r| = 0.34.", Caption),
    ]))

    # 3.2 All 8 vectors
    story.append(Paragraph("3.2 The Collapse Extends to Synonym Vectors", SubHead))
    story.append(Paragraph(
        "To test whether the single-axis finding is specific to the four original trait labels, "
        "we constructed four synonym trait vectors (truthfulness, harm_avoidance, impartiality, "
        "empathy) using the identical pipeline. Projecting all 8 vectors onto ETHICS items "
        "yields an 8×8 <i>projection-score</i> correlation matrix with effective dimensionality "
        "<b>ED = 1.19</b> and PC1 explaining 92% of variance. Every pairwise absolute correlation "
        "exceeds 0.75. We emphasise <i>projection-score</i> here deliberately: Figure 3 reports "
        "Pearson correlations between how the 8 vectors score the same 204 items, which is a "
        "distinct quantity from the cosine similarity between the vectors themselves in activation "
        "space (Table 3 and Figure 3b report that directly). The projection scores can be highly "
        "correlated on this item set even where the underlying vectors are only moderately "
        "cosine-similar — both are true here, and we treat that combination as an interesting "
        "psychometric, dataset-relative result rather than evidence that the vectors are "
        "geometrically collinear.", Body))
    story.append(fig_img(fig_8vec_corr(), width=5.0*inch))
    story.append(Paragraph("Figure 3. 8×8 projection-score correlation matrix (4 original + 4 synonym "
                  "vectors, layer 32; Pearson r of projections across the same 204 ETHICS items). "
                  "White lines separate original from synonym vectors. "
                  "All 8 vectors' projections converge onto essentially the same measurement "
                  "dimension (ED = 1.19) — a statement about projection scores, not vector "
                  "geometry (cf. Figure 3b).", Caption))
    story.append(Spacer(1, 0.06*inch))
    story.append(fig_img(fig_cosine_matrix(), width=4.4*inch))
    story.append(Paragraph("Figure 3b. Pairwise cosine similarity between the 4 original persona "
                  "vectors themselves in activation space (layer 32) — a direct measure of vector "
                  "geometry, independent of any evaluation dataset. Values are far more moderate "
                  "than the projection correlations in Figure 3, showing that near-collinear "
                  "vectors are not required to produce near-redundant projection scores on these "
                  "items.", Caption))

    # Synonym table — Pearson r and cosine columns reload fresh each build from
    # outputs/controls/synonym_vectors/ (regenerated via run_synonym_vector_controls.py,
    # 2026-07-16, after an earlier stale-cache bug — see §4.5). Label AUC is
    # computed directly from data in scripts/verify_paper_numbers.py (previously
    # a hardcoded, unsourced literal — see Unresolved Issues).
    syn_merged = syn_sim.merge(syn_agr[["synonym_id","pearson_r"]], on="synonym_id")
    syn_data = [["Synonym","Parent","Cosine to Parent","ETHICS Pearson r","Label AUC [95% CI]"]]
    for _, r in syn_merged.iterrows():
        pc = f"cosine_{r.parent_trait}"
        auc_r = VERIFIED["synonym_label_auc"][r.synonym_id]
        syn_data.append([r.synonym_id.replace("_"," ").title(), r.parent_trait.capitalize(),
                         f"{r[pc]:.3f}", f"{r.pearson_r:.3f}", ci_str(auc_r)])
    story.append(KeepTogether([
        stbl(syn_data, col_widths=[1.1*inch,0.85*inch,1.0*inch,1.0*inch,1.55*inch]),
        Spacer(1,3),
        Paragraph("Table 3. Synonym vector convergent validity (layer 32). ETHICS projection "
                  "agreement (Pearson r) is <b>high</b>, consistent with Figure 3's shared-axis "
                  "finding — despite this, Label AUC (does the synonym's own projection predict "
                  "the ground-truth ethical label?) stays near chance, mirroring the parent traits' "
                  "own weak label prediction (Table 5). Cosine similarity to the parent vector is "
                  "moderate and does not track projection agreement (e.g. impartiality is weakly "
                  "cosine-similar to fairness yet strongly anti-correlated in projection).", Caption),
    ]))

    # 3.3 Reliability
    story.append(Paragraph("3.3 Reliability: The Shared Axis Is Internally Consistent", SubHead))
    story.append(Paragraph(
        "Although the axis is shared rather than trait-specific, it is <i>reliably</i> measured. "
        "G(k=1) ranges from 0.50 to 0.70; G(k=3) ≥ 0.74 across all trait×layer combinations, "
        "meeting standard psychometric reliability thresholds for paraphrase-to-paraphrase "
        "stability. These G-coefficients reflect only "
        "the stability of each item's score on the shared axis across paraphrastic variants; they "
        "are a reliability result, not a validity result. Paraphrase reliability establishes that "
        "the instrument gives a repeatable reading for the same underlying item — it does not by "
        "itself establish trait specificity, convergent validity, criterion validity, or that the "
        "dimension being measured is the intended one. A measure can be highly reliable while "
        "consistently measuring a shared or unintended construct, which is close to what we find "
        "here: the axis is reliable, but §3.2–§3.4 and §3.9 are what speak to what it is actually "
        "measuring.", Body))

    rd = [["Layer","Trait","G(k=1)","G(k=2)","G(k=3)","G(k=5)"]]
    for _, r in rel_sum.sort_values(["layer","projected_trait"]).iterrows():
        rd.append([str(int(r.layer)), r.projected_trait.capitalize(),
                   f"{r.reliability_1:.3f}", f"{r.reliability_2:.3f}",
                   f"{r.reliability_3:.3f}", f"{r.reliability_5:.3f}"])
    story.append(KeepTogether([
        stbl(rd, col_widths=[0.6*inch,1.1*inch,0.85*inch,0.85*inch,0.85*inch,0.85*inch]),
        Spacer(1,3),
        Paragraph("Table 4. G-coefficients from generalizability theory D-study. "
                  "These measure reliability on the shared axis, not trait specificity.", Caption),
    ]))
    story.append(Spacer(1, 0.06*inch))
    story.append(KeepTogether([
        fig_img(fig_dstudy(), width=5.2*inch),
        Paragraph("Figure 4. D-study curves at layer 32. G(k=3) ≥ 0.74 for all traits, showing "
                  "that scores on the shared dimension are stable across paraphrases. This is a "
                  "reliability result — it shows the measurement is repeatable, not that it is "
                  "trait-specific or that \"moral salience\" is the correct label for what's being "
                  "measured (§4.3).", Caption),
    ]))

    # 3.4 What does the axis capture?
    story.append(Paragraph("3.4 What Does the Shared Axis Capture?", SubHead))
    _pooled = VERIFIED["pooled"]
    story.append(Paragraph(
        "We correlate PC1 scores with ground-truth ethical labels (1=morally wrong, 0=morally OK) "
        "available for all 204 ETHICS items. Overall, PC1 predicts ethical wrongness at "
        f"<b>AUC = {ci_str(_pooled)}</b> (95% CI, stratified bootstrap, n_boot=5000; "
        f"r = {_pooled['pearson_r_with_label']:.3f}) — above chance but weak, and the interval "
        "excludes 0.5 only narrowly. "
        "Breaking this down by item format reveals substantial variation, though per-format "
        "samples are modest (35–74 items) and the resulting intervals are correspondingly wide:", Body))

    fmt_interp = {
        "Justice": "Highest point estimate; justice items have an explicit logical "
                   "entitlement structure, which may make them easier for the axis to track — "
                   "descriptive observation, not a claim about mechanism.",
        "Commonsense": "Weak, imprecise signal — CI is wide and overlaps chance.",
        "EXCUSE": "Near chance; qualitatively many EXCUSE items read as non-sequiturs, "
                  "which may explain the weak signal, though we have not tested this directly.",
        "AITA": "Point estimate below chance (inverted); consistent with — not proof of — the "
                "model favouring well-reasoned posts over the benchmark's verdict, but the CI "
                "is wide enough to include values much closer to chance.",
    }
    CellText = S("CellTxt", fontSize=8, textColor=GREY, leading=10, fontName="Helvetica")
    fmt_data = [["Item Format","n","AUC [95% CI]","Note"]]
    for fmt in ["Justice", "Commonsense", "EXCUSE", "AITA"]:
        r = VERIFIED["by_format"][fmt]
        fmt_data.append([fmt, str(r["n_items"]), ci_str(r), Paragraph(fmt_interp[fmt], CellText)])
    story.append(KeepTogether([
        stbl(fmt_data, col_widths=[0.95*inch,0.35*inch,1.15*inch,3.4*inch]),
        Spacer(1,3),
        Paragraph("Table 5. PC1 predicting ground-truth ethical labels by item format, with "
                  "stratified-bootstrap 95% CIs (5000 resamples, resampling positives/negatives "
                  "separately within each format). Format explains 15.8% of PC1 variance; the "
                  "format–trait confound in ETHICS limits interpretation of what the axis "
                  "represents, and small per-format n means these estimates are imprecise — "
                  "point estimates should not be over-interpreted format-to-format.", Caption),
    ]))
    story.append(Spacer(1, 0.06*inch))
    story.append(KeepTogether([
        fig_img(fig_pc1_label(), width=6.2*inch),
        Paragraph("Figure 5. Left: PC1 score distribution by ethical label. "
                  "Right: AUC by item format (see Table 5 for 95% CIs). Justice items show the "
                  "highest point estimate; AITA items show the lowest (below chance).", Caption),
    ]))
    story.append(Paragraph(
        "The AITA inversion is suggestive, though the CI on that format's AUC is wide enough that "
        "it should be read cautiously: qualitatively, AITA posts that score high on PC1 tend to be "
        "well-reasoned narratives where the person is contextually justified, while the benchmark "
        "label may still classify the action as wrong. One plausible reading is that PC1 may "
        "capture something like <b>moral salience or reasoning intensity</b> — how strongly a "
        "scenario engages ethical reasoning — rather than moral correctness as defined by the "
        "benchmark. We present this as a tentative interpretation consistent with the weak pooled "
        "AUC, the stronger justice-format result, and the inverted AITA result, not as an "
        "established characterisation of what the axis represents.", Body))

    # 3.5 Null controls
    story.append(Paragraph("3.5 Null Distribution Controls", SubHead))
    story.append(Paragraph(
        "Random vector projections confirm the real vectors' reliability is not a geometry "
        "artefact: null G(k=1) < 0.004 vs. real 0.50–0.70. "
        "The shuffled-label specificity test uses two metrics. <b>Diagonal dominance</b> is the "
        "fraction of items where the item's own labelled-trait vector gives the highest projection "
        "among the four (chance = 0.25 for 4 traits). <b>Matching margin</b> is defined per item as "
        "(projection on the item's own labelled-trait vector) minus (the highest projection among "
        "the other three trait vectors), averaged across items — positive values mean the labelled "
        "trait \"wins\" on average, negative values mean some other trait vector scores the item "
        "higher on average than its own labelled trait does. Both metrics are computed on the "
        "mean-centred projections (§2.2), and because centring subtracts a separate constant per "
        "trait column, the <i>absolute</i> value of the matching margin (e.g. roughly −800) is not "
        "itself a meaningful, zero-calibrated quantity — only its value relative to the "
        "permutation null below is interpretable. For both metrics, the p-value is the fraction of "
        "10,000 label-shuffling permutations (item labels randomly reassigned, projection columns "
        "held fixed) whose metric is ≥ the real value — a one-sided test of whether real trait-label "
        "alignment exceeds chance. We reviewed the implementation "
        "(<i>src/controls/shuffled_labels.py</i>) and found the permutation and p-value logic "
        "correct; we keep the name \"matching margin\" (it accurately describes the quantity) but "
        "no longer report its raw value without this explanation. "
        "The shuffled-label specificity test is <b>weakly significant</b>: diagonal dominance "
        "p = 0.0226 (real = 0.275, null 95th pct = 0.265, 10,000 permutations), but matching "
        "margin p = 0.0787 (not significant). This indicates marginal trait-label specificity, "
        "consistent with the near-collapsed structure.", Body))

    shuf_data = [["Metric","Real Value","Null Mean","Null 95th pct","p-value"]]
    for _, r in shuf.iterrows():
        shuf_data.append([r.metric.replace("_"," ").title(),
                          f"{r.real_value:.4f}", f"{r.null_mean:.4f}",
                          f"{r.null_p95:.4f}", f"{r.p_value:.4f}"])
    story.append(KeepTogether([
        stbl(shuf_data, col_widths=[1.5*inch,0.9*inch,0.9*inch,1.0*inch,0.7*inch]),
        Spacer(1,3),
        Paragraph("Table 6. Shuffled-label specificity test (10,000 permutations). Diagonal "
                  "dominance is marginally significant (p=0.0226); matching margin is not "
                  "(p=0.0787). Both are far weaker than mock-data results previously reported.", Caption),
    ]))

    story.append(Paragraph(
        "The pooled diagonal-dominance figure above (27.5%) masks substantial heterogeneity "
        "across ETHICS item formats. Stratifying by ETHICS source split (justice, deontology, "
        "commonsense — recall from §2.3 that the commonsense split itself spans both our "
        "Commonsense- and AITA-format items) "
        "shows that trait-label alignment is concentrated almost entirely in justice-format "
        "items:", Body))

    dd_data = [["Format","n","Diagonal Dominance","p (vs. chance=0.25)"]]
    for fmt, n, dd, p in DIAG_BY_FORMAT:
        dd_data.append([fmt.capitalize(), str(n), f"{dd:.3f}",
                         f"{p:.4f}" if p >= 0.0001 else "<0.0001"])
    story.append(KeepTogether([
        stbl(dd_data, col_widths=[1.3*inch,0.6*inch,1.5*inch,1.6*inch]),
        Spacer(1,3),
        Paragraph("Table 7. Diagonal dominance stratified by ETHICS source split "
                  "(two-sided z-test vs. chance=0.25; note n here uses the 3-way source-split "
                  "grouping, not the 4-way format grouping in Table 5 — see §2.3). Justice items "
                  "show strong, "
                  "significant trait-label alignment — nearly double the pooled estimate. "
                  "Commonsense items are at or below chance. Deontology is uninformative here "
                  "because 96% of its items are honesty-labelled (§2.3), leaving little "
                  "trait variance to test.", Caption),
    ]))
    story.append(Paragraph(
        "This reframes the validity picture from Table 5 above: rather than a uniformly weak "
        "signal across all items, the shared axis carries a real, statistically robust "
        "trait-label signal specifically within justice-format items — consistent with justice "
        "also showing the strongest ground-truth label AUC (0.711). The pooled 'marginal' "
        "significance in Table 6 appears to be diluted by formats (commonsense, deontology) "
        "where the axis carries little or no trait-specific signal.", Body))

    # 3.6 Contrast validation
    story.append(Paragraph("3.6 Internal Construction-Set Separation: Separating Construction-Set Items", SubHead))
    story.append(Paragraph(
        "Persona vectors achieve AUC ≥ 0.80 at layers 32–47 when separating high-persona from "
        "low-persona items drawn from the same contrastive construction distribution. "
        "This is <b>internal construction-set separation</b>, not convergent or criterion "
        "validity — the difference-of-means vector maximally separates its "
        "own training signal by construction — and should not be interpreted as evidence of "
        "generalisation to independent items or as trait specificity. It confirms only that the "
        "pipeline is internally consistent. This is the same category of check as the virtue_axis "
        "AUC reported in §3.9.", Body))

    auc_data = [["Trait","Layer 32","Layer 40","Layer 47"]]
    for t in ["honesty","harmlessness","fairness","compassion"]:
        row = [t.capitalize()]
        for l in [32,40,47]:
            v = contrast[(contrast.trait==t)&(contrast.layer==l)]["auc"].values[0]
            row.append(f"{v:.3f}")
        auc_data.append(row)
    story.append(KeepTogether([
        stbl(auc_data, col_widths=[1.3*inch,1.1*inch,1.1*inch,1.1*inch]),
        Spacer(1,3),
        Paragraph("Table 8. Contrast validation AUC (all ≥ 0.80 at primary layers). "
                  "Note: this tests separation of construction-set items, not generalisation "
                  "to independent benchmarks.", Caption),
    ]))

    story.append(KeepTogether([
        fig_img(fig_contrast(), width=5.2*inch),
        Paragraph("Figure 6. Contrast validation AUC by trait and layer. "
                  "High AUC reflects internal consistency of the construction pipeline.", Caption),
    ]))

    # 3.7 Preprocessing / ceiling
    story.append(Paragraph("3.7 Preprocessing Robustness and Ceiling Check", SubHead))
    story.append(Paragraph(
        "Raw and mean-centred ETHICS projections produce identical structure metrics "
        "(ED = 1.130, mean |r| = 0.919 under both conditions), confirming preprocessing "
        "is not a source of the collapse. Exact-duplicate ceiling checks yield G = 1.00 "
        "for all traits, confirming the pipeline's theoretical upper bound is achievable.", Body))

    # 3.8 Synthetic confound-controlled replication
    story.append(Paragraph("3.8 Confound-Controlled Replication: A Synthetic Item Bank", SubHead))
    story.append(Paragraph(
        "The ETHICS-based analysis above is limited by a structural confound: item <i>format</i> "
        "is nearly collinear with <i>trait</i> (deontology is 96% honesty, justice is 72% "
        "fairness — §2.3). To test whether the single-axis collapse is a property of the model "
        "or an artefact of this confound, we constructed an independent 160-item bank: 4 traits × "
        "20 matched pairs, each pair a single-sentence, first-person scenario that either upholds "
        "or violates the trait, sharing context and differing only in the trait-relevant action "
        "(e.g. <i>told the cashier about extra change</i> vs. <i>kept it silently</i>). Zero "
        "literal trait-name words appear anywhere, checked programmatically. This bank shares no "
        "format, vocabulary, or construction with either ETHICS or the persona-vector construction "
        "prompts.", Body))
    story.append(Paragraph(
        "<b>Construction procedure.</b> All 160 items are hand-authored: the 20 matched pairs per "
        "trait are written directly as literal text in the build script "
        "(<i>scripts/build_synthetic_trait_bank.py</i>), not generated by an LLM call within that "
        "script. Nine drafting rules were followed (documented in full in "
        "<i>docs/synthetic_item_bank_guidelines.md</i>): uniform single-sentence first-person "
        "format; matched upheld/violated pairs sharing context; zero literal trait-name leakage; "
        "single-trait isolation per item; balanced 20/20 labels per trait; mundane rather than "
        "dramatic scenarios; contextual diversity across pairs; independence from the persona-vector "
        "construction prompts; and schema compatibility with the ETHICS pipeline. Primary-trait "
        "membership is assigned by construction — each pair is authored under one of four "
        "trait-specific lists — rather than by a separate classification step. "
        "Trait-name leakage is checked programmatically via regex word-boundary matching against "
        "the exact word list used by the project's own contrastive-prompt audit tool "
        "(<i>src/vectors/artifact_quality.py</i>): e.g. honesty flags \"honest/honesty/truthful/"
        "truth/deceptive/lie/lies\"; compassion flags \"compassion/compassionate/empathy/"
        "empathetic\"; similarly for harmlessness and fairness. A build-time self-check also "
        "asserts exact row counts (160 total, 40/trait, 20/20 label split) and unique item IDs.", Body))
    story.append(Paragraph(
        "<b>Validation limitation.</b> Beyond these programmatic structural and leakage checks, "
        "there is no independent semantic-quality validation step for this bank — no second-rater "
        "review and no LLM-based quality pass, unlike the ETHICS-derived item bank's annotation "
        "pipeline (§2.3) or the contrastive-prompt artifact bank's Stage 2A-review audit (§4.1). "
        "Each item carries a self-declared \"high\" confidence label from the single author who "
        "wrote it. We were unable to determine from the repository whether any items were excluded "
        "or edited during drafting, or whether informal model-assisted drafting preceded the "
        "hand-authored final text now in the build script; neither is recorded. We report this "
        "honestly as a limitation rather than implying an independent validation step that did not "
        "happen.", Body))

    synth_struct_data = [
        ["Layer", "Metric", "ETHICS (204 items)", "Synthetic bank (160 items)"],
        ["32", "Eff. Dim.", "1.130", "1.214"],
        ["32", "Top correlated pair", "harmlessness–fairness", "harmlessness–fairness"],
        ["40", "Eff. Dim.", "2.461", "2.091"],
        ["40", "Top correlated pair", "honesty–fairness", "honesty–fairness"],
        ["47", "Eff. Dim.", "1.151", "1.389"],
    ]
    story.append(KeepTogether([
        stbl(synth_struct_data, col_widths=[0.55*inch,1.55*inch,1.7*inch,1.9*inch]),
        Spacer(1,3),
        Paragraph("Table 9. Structure replication on the synthetic confound-controlled bank. "
                  "The layer-32/47-collapse vs. layer-40-partial-separation pattern replicates "
                  "directionally at every layer; at layers 32 and 40 the identical pair of traits "
                  "is most correlated in both independent datasets.", Caption),
    ]))
    story.append(Paragraph(
        "This is stronger evidence than “collapse happens somewhere”: the <i>same specific "
        "pair</i> of traits collapses together on data sharing nothing with ETHICS, which directly "
        "answers whether the layer-40 separation was a fluke of testing only three layers once — "
        "a pure artefact would not be expected to replicate this precisely.", Body))
    story.append(Paragraph(
        "Within-trait discrimination (does each trait's own vector separate its upheld from its "
        "violated items?) shows a striking split. Fairness discriminates cleanly in the direction "
        "predicted by the vector's construction (upheld scores higher; AUC 0.73–0.86 at layers "
        "32 and 47, robust to removing the most extreme items). Harmlessness and compassion also "
        "discriminate significantly, but in the <i>opposite</i> direction from what their "
        "construction would predict. Honesty shows no discrimination at any layer — notable "
        "because honesty was the dominant, most narratively convenient trait in the ETHICS "
        "analysis (96% of EXCUSE-format items). Projecting the four synonym vectors onto this "
        "same bank replicates both the 8-vector collapse (ED = 1.28–1.68 across layers, close "
        "to the ETHICS-based 1.19) and, for 3 of 4 trait pairs, the same direction of "
        "discrimination as their parent vector — direction agreeing across independently "
        "constructed vectors is stronger convergent-validity evidence than either result alone.", Body))

    # 3.9 Virtue-axis control vector
    story.append(Paragraph("3.9 Testing One Candidate Explanation: A Virtue-Axis Control Vector", SubHead))
    story.append(Paragraph(
        "One plausible explanation for the single-axis collapse is that all four traits' "
        "contrastive prompts secretly encode one generic, RLHF-instilled “aligned vs. "
        "unaligned persona” rather than trait-specific content. We tested one specific "
        "operationalisation of this idea directly by "
        "constructing a control vector (<i>virtue_axis</i>) using the identical Chen et al. "
        "pipeline, but with system prompts deliberately generic — “be a virtuous AI” vs. "
        "“be an unethical AI,” with no mention of honesty, harmlessness, fairness, or "
        "compassion. To isolate the system prompt as the only manipulated variable, the 40 "
        "elicitation questions were not newly written but reused verbatim from the four existing "
        "trait pools (10 per trait, matching every other vector's construction sample size). "
        "We test this one construction; other operationalisations of \"generic alignment\" "
        "(different prompt wording, different elicitation questions, or a non-contrastive "
        "construction method entirely) might behave differently, and our results below speak "
        "only to this specific vector.", Body))
    story.append(Paragraph(
        "The resulting vector separates its own held-out high/low construction items more "
        "cleanly than any of the four trait vectors separate theirs "
        "(AUC 0.898–1.000 across all six candidate layers, reaching 1.000 at layers "
        "28, 32, and 47). We flag explicitly that this AUC is an <b>internal construction-set "
        "validation</b> — it measures whether virtue_axis separates the very high/low-persona "
        "items used to build it, the same near-circular check described for the trait vectors in "
        "§3.6 — and should not be read as independent criterion validity or evidence that "
        "virtue_axis is a good general-purpose \"AI goodness\" detector outside this "
        "construction distribution. Within that scope, a generic good-vs-bad persona is, if "
        "anything, an easier direction for the model to represent cleanly than any specific moral "
        "trait.", Body))

    virt_pc1_data = [
        ["Layer", "cos(virtue_axis, PC1)", "ED: 4 traits", "ED: +virtue_axis (5-vector)", "Δ ED"],
        ["32", "−0.093", "1.130", "1.508", "+0.378"],
        ["40", "−0.056", "2.461", "2.162", "−0.299"],
        ["47", "−0.309", "1.151", "1.662", "+0.510"],
    ]
    story.append(KeepTogether([
        stbl(virt_pc1_data, col_widths=[0.55*inch,1.5*inch,1.15*inch,1.65*inch,0.85*inch]),
        Spacer(1,3),
        Paragraph("Table 10. virtue_axis vs. the shared collapse direction (PC1), tested two "
                  "independent ways. Left: cosine similarity between virtue_axis and PC1 "
                  "reconstructed in activation space from the trait vectors' loadings. Right: "
                  "effective dimensionality when virtue_axis is added as a 5th vector to the "
                  "real ETHICS projection structure analysis.", Caption),
    ]))
    story.append(Paragraph(
        "Both tests agree: <b>virtue_axis is not the shared collapse direction.</b> Cosine "
        "similarity to the reconstructed PC1 is near zero (and slightly negative) at every "
        "primary layer — not the strong positive alignment the generic-alignment hypothesis "
        "predicts. Independently, adding virtue_axis as a 5th vector to the real ETHICS "
        "structure analysis <i>increases</i> effective dimensionality at layers 32 and 47 (the "
        "two layers where the four-trait collapse is strongest) rather than leaving it "
        "unchanged, as a redundant 5th copy of the same shared axis would. Layer 40 is a partial "
        "exception — there, virtue_axis correlates almost perfectly with honesty (r = 0.99) and "
        "fairness (r = 0.98) specifically, and its addition slightly decreases effective "
        "dimensionality — but this is the layer where the original four traits are themselves "
        "least collapsed, not the layers the generic-alignment story needs to explain. Taken "
        "together, this specific, deliberately generic operationalisation of “AI "
        "goodness vs. badness” — despite separating its own construction items cleanly — does "
        "not explain the axis the four moral traits collapse onto. We cannot generalise from one "
        "construction to all possible operationalisations of generic alignment.", Body))

    # ── §4 DISCUSSION ────────────────────────────────────────────────────────
    story.append(Paragraph("4. Discussion", SectHead))

    story.append(Paragraph("4.1 The Single-Axis Finding and its Relationship to Chen et al.", SubHead))
    story.append(Paragraph(
        "Our primary finding — that the contrastive persona vector method extracts a single "
        "shared moral-valence axis rather than four trait-specific directions — formally "
        "quantifies an observation made in passing by Chen et al. (2025). In their footnote 6, "
        "they note that persona shifts across traits 'tend to shift together' and suspect "
        "'correlations between the underlying persona vectors.' Our systematic measurement "
        "confirms this: ED ≈ 1.13 at primary layers, with all 8 original and synonym "
        "vectors projecting onto a single dimension (8-vector ED = 1.19, PC1 = 92%).", Body))
    story.append(Paragraph(
        "One plausible explanation would be that RLHF training has instilled a strong "
        "<i>aligned vs. unaligned persona</i> concept that dominates the residual stream "
        "whenever the model is placed in high- or low-moral-persona mode. The system prompts "
        "for all four traits — regardless of whether they invoke honesty, fairness, or compassion "
        "— all amount to 'be a virtuous AI' vs. 'be an unethical AI,' and on this account the "
        "model would simply be shifting along one dominant, generic alignment axis.", Body))
    story.append(Paragraph(
        "Two pieces of evidence argue against this specific account, rather than merely "
        "failing to confirm it. First, the Stage 2A-review confound audit — designed specifically "
        "to flag prompts that collapse into generic AI-goodness rather than trait-specific "
        "content — found zero warning- or high-severity findings across all 40 system prompts and "
        "160 elicitation questions; the prompts are reasonably trait-specific by this automated "
        "check, yet the vectors still collapsed. Second, and more directly, we constructed a "
        "control vector (<i>virtue_axis</i>, §3.9) from deliberately generic contrastive prompts "
        "carrying exactly the “virtuous vs. unethical AI” framing this hypothesis requires. If the "
        "hypothesis were correct, this vector should align strongly with the shared collapse "
        "direction (PC1). It does not: cosine similarity to PC1 is near zero at every primary "
        "layer (−0.09 to −0.31), and adding it as a 5th vector <i>increases</i> effective "
        "dimensionality at the two layers where the collapse is strongest, rather than leaving it "
        "unchanged as a redundant copy of the same axis would. This one deliberately generic "
        "construction — which separates its own high/low construction items cleanly (an internal "
        "check, not independent criterion validity; §3.9) — fails a direct test of the specific "
        "hypothesis it operationalises. The collapse appears to be real and specific to something "
        "about how the model "
        "represents these four moral constructs together — not simply a stand-in for this "
        "particular operationalisation of "
        "RLHF alignment — though what that something is remains an open question this design "
        "cannot resolve on its own (e.g. it does not rule out a different, more specific shared "
        "direction than the one virtue_axis operationalises, a different generic-alignment "
        "construction behaving differently, or effects specific to this one "
        "model).", Body))

    story.append(Paragraph("4.2 The Layer 40 Anomaly", SubHead))
    story.append(Paragraph(
        "Layer 40 shows meaningfully more separation (ED = 2.46, PC1 = 53.4%, mean |r| = 0.34) "
        "than layers 32 or 47. This non-monotonic pattern — more differentiation in the middle "
        "layers, collapsing again at the final layers — suggests the model does process "
        "trait-relevant information in a more distributed way at intermediate depths, before "
        "this converges to a unified representation. Characterising what the second dimension "
        "at layer 40 captures is a natural direction for future work.", Body))

    story.append(Paragraph("4.3 What the Shared Axis May Measure (A Tentative Interpretation)", SubHead))
    story.append(Paragraph(
        f"PC1 predicts ground-truth ethical labels at AUC = {ci_str(VERIFIED['pooled'])} overall "
        f"(95% CI), rising to {ci_str(VERIFIED['by_format']['Justice'])} "
        f"for justice-format items and falling to {ci_str(VERIFIED['by_format']['AITA'])} "
        "(inverted) for AITA-format items — intervals that are wide given format-level sample "
        "sizes of 35–74 items. "
        "This pattern is <i>consistent with</i> an interpretation in terms of <i>moral salience</i> "
        "— how strongly the model engages ethical reasoning for a given scenario — rather than "
        "moral correctness per se, but the evidence is indirect (a weak pooled AUC plus "
        "format-dependent variation) and this interpretation remains tentative; we have not "
        "directly tested salience as a construct against alternative accounts. "
        "The ETHICS benchmark's format–trait confound (each trait appears predominantly in "
        "one item format) further limits how cleanly this can be interpreted from this dataset "
        "alone.", Body))

    story.append(Paragraph("4.4 Implications and Limitations", SubHead))
    for pt in [
        "<b>The method works but is coarse:</b> Persona vectors reliably extract something real "
        "from the residual stream (non-random, reliable, non-trivially predicts labels). The "
        "limitation is resolution — the method recovers a single moral-valence axis, not "
        "four independent trait measurements.",
        "<b>Monitoring applications remain valid:</b> Chen et al.'s primary use case — "
        "monitoring and steering the model's overall moral alignment — is supported by our "
        "findings. A single reliable axis can still flag deployment-time persona drift.",
        "<b>Trait-specific measurement requires different methods:</b> Orthogonalisation "
        "(projecting out the shared axis before extracting a second direction), supervised "
        "probing on trait-labelled items, or causal steering experiments would be needed "
        "to determine whether trait-specific directions exist at a finer level.",
        "<b>Single model:</b> All results are for Gemma-3-12B-IT. The generality of the "
        "single-axis finding across model families and sizes is unknown.",
        "<b>Benchmark confound:</b> The ETHICS item format–trait confound means some apparent "
        "trait effects may be format effects. A balanced benchmark with controlled format is "
        "needed for cleaner analysis.",
    ]:
        story.append(Paragraph(f"• {pt}", Bullet))

    story.append(Paragraph("4.5 Reproducibility and Data Integrity", SubHead))
    story.append(Paragraph(
        "An initial structure analysis run in June used stale mock activations (dim=64) written "
        "as a pipeline placeholder before real GPU extraction was available. That mock run "
        "produced ED ≈ 3.87, an artefact of random projections onto 4 unit vectors being "
        "naturally near-orthogonal in high dimension — not a finding about the model. All results "
        "reported in this paper use the corrected real Gemma-3-12B-IT activations (re-extracted "
        "June 29, dim = 3840); the mock result has been removed from every table, figure, and "
        "cached intermediate file used by the build. Separately, during this revision we found "
        "and corrected one genuinely stale cached output: the synonym vector convergent-validity "
        "control (Table 3) had been computed on 2026-06-29 against an earlier version of the "
        "ETHICS projection data that was itself regenerated on 2026-07-08; the Pearson-r and "
        "Label-AUC columns in Table 3 have been recomputed against current data (see §3.2) and "
        "now agree with the independently-computed 8-vector correlation matrix in Figure 3.", Body))

    # ── §5 CONCLUSION ────────────────────────────────────────────────────────
    story.append(Paragraph("5. Conclusion", SectHead))
    story.append(Paragraph(
        "We applied the Chen et al. (2025) persona vector method to four moral character traits "
        "in Gemma-3-12B-IT and conducted, to our knowledge, the first systematic psychometric "
        "evaluation of the resulting measurement structure. Our central finding is that the "
        "method reliably produces a single shared moral-valence measurement dimension rather "
        "than four trait-specific ones: effective dimensionality ≈ 1.13 at layers 32 and 47, "
        "with all 8 original and synonym vectors' projections converging onto this dimension. "
        "This is a claim about the measurement structure the vectors produce on these items, not "
        "a claim that the vectors themselves are geometrically collinear (§3.2 reports direct "
        "cosine similarities, which are more moderate). The axis is reliably measured "
        "(G(k=3) ≥ 0.74) and predicts ethical labels at AUC = 0.585 (95% CI [0.51, 0.66]), "
        "with format-dependent "
        "variation consistent with — though not proof of — an interpretation in terms of moral "
        "salience rather than moral correctness; this interpretation remains tentative (§4.3).", Body))
    story.append(Paragraph(
        "Two further results strengthen and sharpen this picture. First, the collapse is not an "
        "artefact of the ETHICS benchmark's item-format confound specifically: an independent, "
        "format-controlled 160-item bank, built to remove that particular confound, reproduces the "
        "same structure almost exactly, down to the same most-correlated trait pair at the same "
        "layers — though it cannot rule out other lexical, semantic, or generation-related "
        "confounds we have not identified. Within that bank, the four traits also show "
        "distinct, non-uniform behaviour — fairness discriminates cleanly in the expected "
        "direction, harmlessness and compassion discriminate but backwards, and honesty shows no "
        "signal at all — indicating the underlying structure is not simply undifferentiated noise. "
        "Second, we directly tested one natural candidate explanation for the collapse — a generic "
        "RLHF-instilled aligned-vs-unaligned persona, as operationalised by our specific "
        "virtue_axis construction — by building a control vector from deliberately generic "
        "contrastive prompts. This vector separates its own high/low construction items more "
        "cleanly than any trait vector (an internal validation of that construction distribution, "
        "not independent criterion validity), yet is nearly orthogonal to the actual shared "
        "collapse direction and, at the layers where the collapse is strongest, its addition "
        "increases rather than leaves unchanged the projection structure's effective "
        "dimensionality.", Body))
    story.append(Paragraph(
        "These findings formally quantify the cross-trait correlation pattern flagged by "
        "Chen et al., show that the collapse is not simply an artefact of the ETHICS format–trait "
        "confound, and show that one natural candidate explanation — this specific virtue_axis "
        "operationalisation of generic alignment — does not survive a direct empirical test "
        "designed to confirm it (other generic or multidimensional alignment representations are "
        "not ruled out). This is a more informative outcome than either a clean confirmation or a "
        "simple null result: the "
        "contrastive difference-of-means method is measuring something real and reliable about "
        "this model's representation of moral character, but that something is neither four "
        "trait-specific directions nor a simple stand-in for this particular operationalisation of "
        "generic alignment. Whether "
        "finer-grained trait directions exist in the model — and whether orthogonalisation, "
        "supervised probing, or comparison across model families and RLHF regimes could recover "
        "them — remains an open and now more precisely specified question for AI psychometrics "
        "and interpretability research.", Body))

    # ── REFERENCES ───────────────────────────────────────────────────────────
    story.append(Paragraph("References", SectHead))
    refs = [
        "Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona vectors: Monitoring and controlling character traits in language models. <i>arXiv:2507.21509</i>.",
        "Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering latent knowledge in language models without supervision. <i>ICLR 2023</i>.",
        "Elhage, N., Nanda, N., Olsson, C., et al. (2022). A mathematical framework for transformer circuits. <i>Transformer Circuits Thread</i>.",
        "Hendrycks, D., Burns, C., Basart, S., et al. (2021). Aligning AI with shared human values. <i>ICLR 2021</i>.",
        "Turner, A. M., Thiergart, L., Udell, G., et al. (2023). Activation addition: Steering language models without optimization. <i>arXiv:2308.10248</i>.",
        "Zou, A., Phan, L., Chen, S., et al. (2023). Representation engineering: A top-down approach to AI transparency. <i>arXiv:2310.01405</i>.",
        "Gemma Team, Google DeepMind. (2025). Gemma 3 technical report. <i>arXiv:2503.19786</i>.",
        "Brennan, R. L. (2001). <i>Generalizability Theory</i>. Springer.",
    ]
    for i, r in enumerate(refs):
        story.append(Paragraph(f"[{i+1}] {r}", S("Ref", parent=Body, fontSize=8.5, spaceAfter=4)))

    doc.build(story)
    for p in _tmp_pngs:
        try: os.unlink(p)
        except: pass
    print(f"Saved: {OUT}")

if __name__ == "__main__":
    build()
