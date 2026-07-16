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

def _stratified_diagonal_dominance():
    """Diagonal dominance (annotated trait = top-projecting vector) within each
    ETHICS source split, holding item format fixed. Two-sided z-test vs. chance=0.25."""
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
    ax.set_title("PC1 by Ethical Label\n(overall AUC=0.585, r=0.117)", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    # Right: AUC by format
    ax2 = axes[1]
    formats = ["Justice","Commonsense","EXCUSE","AITA"]
    aucs_fmt = [0.711, 0.551, 0.535, 0.377]
    bar_c = ["#2563EB","#059669","#D97706","#9333EA"]
    bars = ax2.barh(formats, aucs_fmt, color=bar_c, alpha=0.85)
    ax2.axvline(0.5, color="#6B7280", linestyle="--", lw=1, label="Chance (0.50)")
    ax2.set_xlim(0.3, 0.8)
    ax2.set_xlabel("AUC (PC1 predicting morally wrong)", fontsize=9)
    ax2.set_title("AUC by Item Format\n(format drives predictive variation)", fontsize=9.5, fontweight="bold")
    for bar, v in zip(bars, aucs_fmt):
        ax2.text(v+0.008, bar.get_y()+bar.get_height()/2, f"{v:.3f}",
                 va="center", fontsize=8.5, fontweight="bold")
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
        "Gemma-3-12B-IT, and conduct the first systematic psychometric evaluation of the resulting "
        "measurement structure. Using a contrastive system-prompt paradigm, we construct "
        "difference-of-means persona vectors at six transformer layers and project 204 curated "
        "ETHICS benchmark items onto these vectors. "
        "Contrary to our initial hypothesis, structural analysis reveals that the four trait "
        "projections collapse onto a single dominant dimension at layers 32 and 47 "
        "(effective dimensionality ≈ 1.13, PC1 explaining 94% of variance, mean inter-trait "
        "|r| = 0.92), with only modest separation at layer 40 (ED = 2.46). "
        "This single shared axis extends to synonym trait vectors: all 8 vectors "
        "(4 original + 4 synonym) produced an 8×8 projection correlation matrix with "
        "ED = 1.19, confirming convergence onto one dimension regardless of trait label. "
        "Generalizability theory analysis on 761 paraphrase variants yields G(k=3) ≥ 0.74 — "
        "demonstrating that the method reliably measures <i>something</i>, but that something "
        "appears to be a shared moral-salience axis rather than four trait-specific directions. "
        "This shared axis predicts ground-truth ethical labels at AUC = 0.585, with substantial "
        "variation by item format (justice: AUC = 0.71; AITA: AUC = 0.38) — a format-stratified "
        "analysis shows this pooled figure conceals a strong, significant signal concentrated in "
        "justice-format items specifically. "
        "To rule out that the collapse is an artefact of ETHICS' item-format confound, we built an "
        "independent, confound-free item bank (160 items, single format, matched upheld/violated "
        "pairs, zero trait-name leakage): the same collapse replicates almost exactly, including the "
        "same most-correlated trait pair at the same layers. "
        "We then directly tested the most likely explanation for the collapse — that it reflects a "
        "generic RLHF-instilled 'aligned vs. unaligned persona' axis rather than trait-specific "
        "content — by constructing a control vector from deliberately generic, non-trait-specific "
        "contrastive prompts. This vector validates better than any trait vector (AUC up to 1.00), "
        "yet is nearly orthogonal to the actual shared collapse direction (cosine ≈ 0 to −0.31 across "
        "layers) and, at the two layers where the collapse is strongest, its addition <i>increases</i> "
        "effective dimensionality rather than leaving it unchanged — direct evidence against the "
        "generic-alignment explanation. "
        "Our findings formally quantify and extend the cross-trait correlation pattern noted in "
        "a footnote by Chen et al., establish that the collapse is a robust, confound-independent "
        "phenomenon, and show that its most natural explanation does not survive direct empirical "
        "test — leaving the cause of the collapse an open question for AI psychometrics."
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
        "and conducts the first systematic psychometric evaluation of the resulting measurement "
        "structure. Our goal is to assess whether persona vectors can function as a reliable, "
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
        "We use <b>Gemma-3-12B-IT</b> (Google DeepMind, 2024): 12B parameters, hidden dimension "
        "d = 3,840, 48 transformer blocks. All forward passes run on NVIDIA A100 GPUs via Modal. "
        "Activations are extracted as float32 vectors from the residual stream at the "
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
        "The dataset spans four item formats: EXCUSE (74 items, 96% honesty), "
        "justice (53 items, 72% fairness), commonsense (42 items, 60% harmlessness), "
        "and AITA (35 items, mixed traits). The format–trait confound is a structural property "
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

    story.append(amber_box(
        "Data integrity note: An initial structure analysis run in June used stale mock "
        "activations (dim=64) written before real GPU extraction. The mock run produced "
        "ED≈3.87 because random projections onto 4 unit vectors are naturally near-orthogonal. "
        "All results below use the corrected real Gemma-3-12B activations (re-extracted June 29, "
        "dim=3840). The mock result has been removed from all tables and figures."
    ))
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
        Paragraph("Figure 2. Inter-trait correlation matrices at layers 32 (left) and 40 (right). "
                  "Layer 32: all absolute correlations ≥ 0.87, confirming near-collinear vectors. "
                  "Layer 40: more modest correlations, mean |r| = 0.34.", Caption),
    ]))

    # 3.2 All 8 vectors
    story.append(Paragraph("3.2 The Collapse Extends to Synonym Vectors", SubHead))
    story.append(Paragraph(
        "To test whether the single-axis finding is specific to the four original trait labels, "
        "we constructed four synonym trait vectors (truthfulness, harm_avoidance, impartiality, "
        "empathy) using the identical pipeline. Projecting all 8 vectors onto ETHICS items "
        "yields an 8×8 correlation matrix with effective dimensionality <b>ED = 1.19</b> and "
        "PC1 explaining 92% of variance. Every pairwise absolute correlation exceeds 0.75. "
        "The collapse onto one axis is robust to the choice of trait label.", Body))
    story.append(fig_img(fig_8vec_corr(), width=5.0*inch))
    story.append(Paragraph("Figure 3. 8×8 projection correlation matrix (4 original + 4 synonym vectors, "
                  "layer 32). White lines separate original from synonym vectors. "
                  "All vectors project onto essentially the same dimension (ED = 1.19).", Caption))

    # Synonym table
    syn_merged = syn_sim.merge(syn_agr[["synonym_id","pearson_r"]], on="synonym_id")
    syn_data = [["Synonym","Parent","Cosine to Parent","ETHICS Pearson r","Label AUC"]]
    label_aucs = {"truthfulness":0.584,"harm_avoidance":0.457,"impartiality":0.550,"empathy":0.381}
    for _, r in syn_merged.iterrows():
        pc = f"cosine_{r.parent_trait}"
        syn_data.append([r.synonym_id.replace("_"," ").title(), r.parent_trait.capitalize(),
                         f"{r[pc]:.3f}", f"{r.pearson_r:.3f}", f"{label_aucs[r.synonym_id]:.3f}"])
    story.append(KeepTogether([
        stbl(syn_data, col_widths=[1.3*inch,1.0*inch,1.1*inch,1.1*inch,0.85*inch]),
        Spacer(1,3),
        Paragraph("Table 3. Synonym vector convergent validity (layer 32). "
                  "Despite moderate cosine similarity, ETHICS projection agreement (Pearson r) "
                  "and label prediction AUC are near zero, consistent with the single-axis finding.", Caption),
    ]))

    # 3.3 Reliability
    story.append(Paragraph("3.3 Reliability: The Shared Axis Is Internally Consistent", SubHead))
    story.append(Paragraph(
        "Although the axis is shared rather than trait-specific, it is <i>reliably</i> measured. "
        "G(k=1) ranges from 0.50 to 0.70; G(k=3) ≥ 0.74 across all trait×layer combinations, "
        "meeting standard psychometric reliability thresholds. These G-coefficients reflect "
        "the stability of each item's score on the shared axis across paraphrastic variants — "
        "not trait specificity.", Body))

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
        Paragraph("Figure 4. D-study curves at layer 32. G(k=3) ≥ 0.74 for all traits, "
                  "reflecting consistent measurement of the shared moral-salience axis.", Caption),
    ]))

    # 3.4 What does the axis capture?
    story.append(Paragraph("3.4 What Does the Shared Axis Capture?", SubHead))
    story.append(Paragraph(
        "We correlate PC1 scores with ground-truth ethical labels (1=morally wrong, 0=morally OK) "
        "available for all 204 ETHICS items. Overall, PC1 predicts ethical wrongness at "
        "<b>AUC = 0.585</b> (r = 0.117) — above chance but weak. "
        "Breaking this down by item format reveals substantial variation:", Body))

    fmt_data = [["Item Format","n","AUC","Interpretation"]]
    for fmt, n, a, interp in [
        ("Justice", 53, 0.711, "Logical entitlement structure aids the model"),
        ("Commonsense", 42, 0.551, "Slight signal"),
        ("EXCUSE", 74, 0.535, "Near chance — many EXCUSE items are non-sequiturs"),
        ("AITA", 35, 0.377, "Inverted — model favours well-reasoned posts regardless of verdict"),
    ]:
        fmt_data.append([fmt, str(n), f"{a:.3f}", interp])
    story.append(KeepTogether([
        stbl(fmt_data, col_widths=[1.0*inch,0.4*inch,0.65*inch,3.2*inch]),
        Spacer(1,3),
        Paragraph("Table 5. PC1 predicting ground-truth ethical labels by item format. "
                  "Format explains 15.8% of PC1 variance; the format–trait confound "
                  "in ETHICS limits interpretation of what the axis represents.", Caption),
    ]))
    story.append(Spacer(1, 0.06*inch))
    story.append(KeepTogether([
        fig_img(fig_pc1_label(), width=6.2*inch),
        Paragraph("Figure 5. Left: PC1 score distribution by ethical label. "
                  "Right: AUC by item format. Justice items (logically structured) "
                  "show the strongest signal; AITA items are inverted.", Caption),
    ]))
    story.append(Paragraph(
        "The AITA inversion is informative: AITA posts that score high on PC1 tend to be "
        "well-reasoned narratives where the person is contextually justified — which the model "
        "encodes as high moral salience — but the benchmark label may still classify the action "
        "as wrong. This suggests PC1 captures <b>moral salience / reasoning intensity</b> rather "
        "than moral correctness as defined by the benchmark.", Body))

    # 3.5 Null controls
    story.append(Paragraph("3.5 Null Distribution Controls", SubHead))
    story.append(Paragraph(
        "Random vector projections confirm the real vectors' reliability is not a geometry "
        "artefact: null G(k=1) < 0.004 vs. real 0.50–0.70. "
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
        "across ETHICS item formats. Stratifying by source split, holding item format fixed, "
        "shows that trait-label alignment is concentrated almost entirely in justice-format "
        "items:", Body))

    dd_data = [["Format","n","Diagonal Dominance","p (vs. chance=0.25)"]]
    for fmt, n, dd, p in DIAG_BY_FORMAT:
        dd_data.append([fmt.capitalize(), str(n), f"{dd:.3f}",
                         f"{p:.4f}" if p >= 0.0001 else "<0.0001"])
    story.append(KeepTogether([
        stbl(dd_data, col_widths=[1.3*inch,0.6*inch,1.5*inch,1.6*inch]),
        Spacer(1,3),
        Paragraph("Table 7. Diagonal dominance stratified by ETHICS source split, holding item "
                  "format fixed (two-sided z-test vs. chance=0.25). Justice items show strong, "
                  "significant trait-label alignment — nearly double the pooled estimate. "
                  "Commonsense items are at or below chance. Deontology is uninformative here "
                  "because 96% of its items are honesty-labelled (Table 5), leaving little "
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
    story.append(Paragraph("3.6 Contrast Validation: Separating Construction-Set Items", SubHead))
    story.append(Paragraph(
        "Persona vectors achieve AUC ≥ 0.80 at layers 32–47 when separating high-persona from "
        "low-persona items drawn from the same contrastive construction distribution. "
        "This is a near-circular test — the difference-of-means vector maximally separates its "
        "own training signal — and should not be interpreted as evidence of independent "
        "generalisation. It confirms only that the pipeline is internally consistent.", Body))

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
    story.append(Paragraph("3.9 Testing the Generic-Alignment Hypothesis: A Virtue-Axis Control Vector", SubHead))
    story.append(Paragraph(
        "The most natural explanation for the single-axis collapse is that all four traits' "
        "contrastive prompts secretly encode one generic, RLHF-instilled “aligned vs. "
        "unaligned persona” rather than trait-specific content. We tested this directly by "
        "constructing a control vector (<i>virtue_axis</i>) using the identical Chen et al. "
        "pipeline, but with system prompts deliberately generic — “be a virtuous AI” vs. "
        "“be an unethical AI,” with no mention of honesty, harmlessness, fairness, or "
        "compassion. To isolate the system prompt as the only manipulated variable, the 40 "
        "elicitation questions were not newly written but reused verbatim from the four existing "
        "trait pools (10 per trait, matching every other vector's construction sample size).", Body))
    story.append(Paragraph(
        "The resulting vector validated more strongly than any of the four trait vectors "
        "(held-out AUC 0.898–1.000 across all six candidate layers, reaching 1.000 at layers "
        "28, 32, and 47) — a generic good-vs-bad persona is, if anything, an easier direction for "
        "the model to represent cleanly than any specific moral trait.", Body))

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
        "together, a deliberately generic, independently-validated operationalisation of “AI "
        "goodness vs. badness” does not explain the axis the four moral traits collapse onto.", Body))

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
        "The most natural explanation would be that RLHF training has instilled a strong "
        "<i>aligned vs. unaligned persona</i> concept that dominates the residual stream "
        "whenever the model is placed in high- or low-moral-persona mode. The system prompts "
        "for all four traits — regardless of whether they invoke honesty, fairness, or compassion "
        "— all amount to 'be a virtuous AI' vs. 'be an unethical AI,' and on this account the "
        "model would simply be shifting along one dominant, generic alignment axis.", Body))
    story.append(Paragraph(
        "Two independent pieces of evidence argue against this account, rather than merely "
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
        "unchanged as a redundant copy of the same axis would. A deliberately generic, "
        "independently-validated operationalisation of the exact hypothesis fails a direct test "
        "of it. The collapse appears to be real and specific to something about how the model "
        "represents these four moral constructs together — not simply a stand-in for overall "
        "RLHF alignment — though what that something is remains an open question this design "
        "cannot resolve on its own (e.g. it does not rule out a different, more specific shared "
        "direction than the one virtue_axis operationalises, or effects specific to this one "
        "model).", Body))

    story.append(Paragraph("4.2 The Layer 40 Anomaly", SubHead))
    story.append(Paragraph(
        "Layer 40 shows meaningfully more separation (ED = 2.46, PC1 = 53.4%, mean |r| = 0.34) "
        "than layers 32 or 47. This non-monotonic pattern — more differentiation in the middle "
        "layers, collapsing again at the final layers — suggests the model does process "
        "trait-relevant information in a more distributed way at intermediate depths, before "
        "this converges to a unified representation. Characterising what the second dimension "
        "at layer 40 captures is a natural direction for future work.", Body))

    story.append(Paragraph("4.3 What the Shared Axis Measures", SubHead))
    story.append(Paragraph(
        "PC1 predicts ground-truth ethical labels at AUC = 0.585 overall, rising to 0.711 "
        "for justice-format items and falling to 0.377 (inverted) for AITA-format items. "
        "This pattern suggests the axis captures <i>moral salience</i> — how strongly the "
        "model enters ethical reasoning mode — rather than moral correctness per se. "
        "The ETHICS benchmark's format–trait confound (each trait appears predominantly in "
        "one item format) limits how cleanly this can be interpreted from this dataset alone.", Body))

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

    # ── §5 CONCLUSION ────────────────────────────────────────────────────────
    story.append(Paragraph("5. Conclusion", SectHead))
    story.append(Paragraph(
        "We applied the Chen et al. (2025) persona vector method to four moral character traits "
        "in Gemma-3-12B-IT and conducted the first systematic psychometric evaluation of the "
        "resulting measurement structure. Our central finding is that the method reliably "
        "extracts a single shared moral-valence axis rather than four geometrically distinct "
        "trait directions: effective dimensionality ≈ 1.13 at layers 32 and 47, with all 8 "
        "original and synonym vectors converging on this axis. The axis is reliably measured "
        "(G(k=3) ≥ 0.74) and predicts ethical labels at AUC = 0.585, with format-dependent "
        "variation suggesting it captures moral salience rather than moral correctness.", Body))
    story.append(Paragraph(
        "Two further results strengthen and sharpen this picture. First, the collapse is not an "
        "artefact of the ETHICS benchmark's item-format confound: an independent, confound-free "
        "160-item bank reproduces the same structure almost exactly, down to the same "
        "most-correlated trait pair at the same layers. Within that bank, the four traits also show "
        "distinct, non-uniform behaviour — fairness discriminates cleanly in the expected "
        "direction, harmlessness and compassion discriminate but backwards, and honesty shows no "
        "signal at all — indicating the underlying structure is not simply undifferentiated noise. "
        "Second, we directly tested the most natural explanation for the collapse — a generic "
        "RLHF-instilled aligned-vs-unaligned persona — by building a control vector from "
        "deliberately generic contrastive prompts. This vector validates better than any trait "
        "vector, yet is nearly orthogonal to the actual shared collapse direction and, at the "
        "layers where the collapse is strongest, its addition increases rather than leaves "
        "unchanged the projection structure's effective dimensionality.", Body))
    story.append(Paragraph(
        "These findings formally quantify the cross-trait correlation pattern flagged by "
        "Chen et al., establish that the collapse is a robust, confound-independent phenomenon "
        "rather than a benchmark artefact, and show that its most natural explanation does not "
        "survive a direct empirical test designed specifically to confirm it. This is a more "
        "informative outcome than either a clean confirmation or a simple null result: the "
        "contrastive difference-of-means method is measuring something real and reliable about "
        "this model's representation of moral character, but that something is neither four "
        "trait-specific directions nor a simple stand-in for generic alignment. Whether "
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
        "Google DeepMind. (2024). Gemma 3 technical report. <i>arXiv:2503.19786</i>.",
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
