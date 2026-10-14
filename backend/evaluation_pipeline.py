"""
RadVision AI — Evaluation Pipeline
Indiana University Chest X-Ray Dataset (IU X-Ray)

Run from your backend folder:
    python evaluation_pipeline.py --location gradcam --tag with_gradcam
    python evaluation_pipeline.py --location prior   --tag without_gradcam
    python ../analysis/statistics.py <without_gradcam per_image csv> <with_gradcam per_image csv>

--location controls the GradCAM ablation:
    gradcam : top-5 markers from GradCAM; marker locations are given to the LLM
    prior   : GradCAM switched off; anatomical prior locations given to the LLM
    none    : GradCAM switched off; no locations in the prompt

Results saved to OUTPUT_DIR (see config below).
"""

import os, sys, json, time, warnings, argparse, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Add backend to path so we can import xray_model and report ──────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these paths if needed
# ══════════════════════════════════════════════════════════════════════════════

# Paths can also be set with environment variables IU_XRAY_DIR / EVAL_OUTPUT_DIR
# or the --data-dir / --output-dir command-line options.
IU_XRAY_DIR = os.getenv("IU_XRAY_DIR", r"C:\Users\gaura\OneDrive\Desktop\xray_project\iu_xray")
IMAGE_DIR   = os.path.join(IU_XRAY_DIR, "images", "images_normalized")
PROJ_CSV    = os.path.join(IU_XRAY_DIR, "indiana_projections.csv")
REPORT_CSV  = os.path.join(IU_XRAY_DIR, "indiana_reports.csv")
OUTPUT_DIR  = os.getenv("EVAL_OUTPUT_DIR", os.path.join(os.path.dirname(BACKEND_DIR), "evaluation_results_v2"))

# Random seed for the 200-image sample (simple random sampling).
# 42 is the value used for the original May 2026 runs.
SEED = 42

# Decision threshold for sensitivity / specificity / precision
THRESHOLD = 0.30

# LLM temperature for evaluation runs (0 = reproducible)
LLM_TEMPERATURE = 0

# How many images to evaluate
# 200  → ~15 minutes, good for testing
# 500  → ~40 minutes, good for paper
# None → full dataset (~3818 images, ~5 hours)
MAX_IMAGES = 200

# Set True to also generate Claude reports and compute BLEU/ROUGE
# Each call costs ~$0.01 — 200 images ≈ $2.00 per run
RUN_REPORT_EVAL = True

# Only evaluate frontal (PA) views — CheXNet is trained on these
FRONTAL_ONLY = True

# ══════════════════════════════════════════════════════════════════════════════
#  DISEASE DEFINITIONS & MeSH MAPPING
# ══════════════════════════════════════════════════════════════════════════════

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# MeSH term → CheXNet disease name (lowercase keys for matching)
MESH_TO_DISEASE = {
    'atelectasis':           'Atelectasis',
    'atelectases':           'Atelectasis',
    'cardiomegaly':          'Cardiomegaly',
    'cardiac enlargement':   'Cardiomegaly',
    'effusion':              'Effusion',
    'pleural effusion':      'Effusion',
    'pleural fluid':         'Effusion',
    'infiltrate':            'Infiltration',
    'infiltration':          'Infiltration',
    'airspace disease':      'Infiltration',
    'airspace opacity':      'Infiltration',
    'interstitial':          'Infiltration',
    'mass':                  'Mass',
    'lung mass':             'Mass',
    'opacity':               'Mass',
    'nodule':                'Nodule',
    'pulmonary nodule':      'Nodule',
    'nodules':               'Nodule',
    'pneumonia':             'Pneumonia',
    'pneumonias':            'Pneumonia',
    'pneumothorax':          'Pneumothorax',
    'consolidation':         'Consolidation',
    'edema':                 'Edema',
    'pulmonary edema':       'Edema',
    'emphysema':             'Emphysema',
    'hyperinflation':        'Emphysema',
    'fibrosis':              'Fibrosis',
    'pulmonary fibrosis':    'Fibrosis',
    'scarring':              'Fibrosis',
    'pleural thickening':    'Pleural_Thickening',
    'pleural disease':       'Pleural_Thickening',
    'hernia':                'Hernia',
    'hiatal hernia':         'Hernia',
    'diaphragmatic hernia':  'Hernia',
}


def parse_mesh(mesh_str):
    """Parse MeSH field → set of matched CheXNet disease names."""
    if pd.isna(mesh_str):
        return set()
    matched = set()
    raw = str(mesh_str).lower()
    # Split on pipe or comma
    terms = [t.strip() for t in raw.replace('|', ',').split(',')]
    for term in terms:
        if term in ('normal', 'no findings', 'unremarkable', ''):
            continue
        # Direct lookup
        if term in MESH_TO_DISEASE:
            matched.add(MESH_TO_DISEASE[term])
            continue
        # Substring lookup
        for key, val in MESH_TO_DISEASE.items():
            if key in term:
                matched.add(val)
                break
    return matched


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE BLEU (no NLTK dependency)
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text):
    """Lower-case word tokens; punctuation removed (same for reference and hypothesis)."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def simple_bleu(reference, hypothesis, n=2):
    """Standard sentence-level BLEU-n (Papineni et al., 2002), no NLTK dependency.

    Uniform weights 1/n over clipped 1..n-gram precisions, brevity penalty
    BP = 1 if c > r else exp(1 - r/c). No smoothing: if any p_k is zero the
    score is zero, as in the original definition.
    """
    from collections import Counter
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    if not hyp or not ref:
        return 0.0
    log_p = 0.0
    for k in range(1, n + 1):
        ref_ng = Counter(tuple(ref[i:i+k]) for i in range(len(ref) - k + 1))
        hyp_ng = Counter(tuple(hyp[i:i+k]) for i in range(len(hyp) - k + 1))
        total = sum(hyp_ng.values())
        if total == 0:
            return 0.0
        clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        if clipped == 0:
            return 0.0
        log_p += np.log(clipped / total) / n
    c, r = len(hyp), len(ref)
    bp = 1.0 if c > r else np.exp(1 - r / c)
    return float(bp * np.exp(log_p))


def rouge_l(reference, hypothesis):
    """Compute ROUGE-L F1 (longest common subsequence)."""
    ref  = tokenize(reference)
    hyp  = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    # LCS dynamic programming
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if ref[i-1]==hyp[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / n if n else 0
    rec  = lcs / m if m else 0
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = ['#1a1a2e','#16213e','#0f3460','#533483','#05c6a2',
          '#e94560','#f5a623','#4ecdc4','#a8e6cf','#dcedc1',
          '#ffd3b6','#ffaaa5','#ff8b94','#c5b9f6']

def plot_roc_curves(all_preds, all_labels, outpath):
    from sklearn.metrics import roc_curve, auc as sk_auc
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    mean_aucs = []
    for i, disease in enumerate(DISEASES):
        gt  = [l[disease] for l in all_labels]
        pr  = [p[disease] for p in all_preds]
        pos = sum(gt)
        if pos < 2:
            continue
        fpr, tpr, _ = roc_curve(gt, pr)
        a = sk_auc(fpr, tpr)
        mean_aucs.append(a)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)],
                lw=1.5, alpha=0.85,
                label=f'{disease} (AUC={a:.3f})')

    ax.plot([0,1],[0,1],'k--', lw=0.8, alpha=0.5, label='Chance')
    ax.set_xlabel('1 - Specificity (FPR)', fontsize=12)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=12)
    ax.set_title('ROC Curves — Per-Disease Classification Performance\n'
                 f'(n = {len(all_labels)} images, mean AUC = {np.mean(mean_aucs):.3f})',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9,
              ncol=2, edgecolor='#ddd')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    return np.mean(mean_aucs)


def plot_auc_bar(metrics_df, outpath):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    ev       = metrics_df.dropna(subset=['AUC'])      # evaluable conditions only
    diseases = ev['Disease'].tolist()
    aucs     = ev['AUC'].tolist()
    colors   = ['#2ecc71' if a >= 0.85 else '#f39c12' if a >= 0.75 else '#e74c3c'
                for a in aucs]

    bars = ax.bar(diseases, aucs, color=colors, edgecolor='white',
                  linewidth=0.8, zorder=3)
    ax.axhline(0.80, color='#2c3e50', linestyle='--', lw=1.5,
               label='AUC = 0.80 threshold', zorder=4)
    ax.axhline(np.mean(aucs), color='#8e44ad', linestyle='-.',
               lw=1.5, label=f'Mean AUC = {np.mean(aucs):.3f}', zorder=4)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='#2c3e50')

    legend_handles = [
        Patch(color='#2ecc71', label='AUC ≥ 0.85 (high)'),
        Patch(color='#f39c12', label='AUC ≥ 0.75 (medium)'),
        Patch(color='#e74c3c', label='AUC < 0.75 (low)'),
    ]
    legend_handles += [h for h in ax.get_legend_handles_labels()[0]]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    ax.set_xlabel('Pathology', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Per-Disease AUC — RadVision AI on IU Chest X-Ray Dataset',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0.4, 1.06])
    ax.set_xticklabels(diseases, rotation=35, ha='right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sensitivity_specificity(metrics_df, outpath):
    metrics_df = metrics_df.dropna(subset=['AUC'])   # evaluable conditions only
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    diseases = metrics_df['Disease'].tolist()
    x = np.arange(len(diseases))
    w = 0.35

    ax.bar(x - w/2, metrics_df['Sensitivity'], w, color='#3498db',
           label='Sensitivity', edgecolor='white', zorder=3)
    ax.bar(x + w/2, metrics_df['Specificity'], w, color='#e67e22',
           label='Specificity', edgecolor='white', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(diseases, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Sensitivity vs Specificity per Disease — RadVision AI',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_disease_prevalence(all_labels, outpath):
    counts = {d: sum(l[d] for l in all_labels) for d in DISEASES}
    n = len(all_labels)
    diseases = list(counts.keys())
    prevalences = [counts[d]/n*100 for d in diseases]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    bars = ax.bar(diseases, prevalences, color='#9b59b6',
                  edgecolor='white', linewidth=0.8, zorder=3)
    for bar, pct in zip(bars, prevalences):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8,
                color='#2c3e50')

    ax.set_xlabel('Pathology', fontsize=12)
    ax.set_ylabel('Prevalence (%)', fontsize=12)
    ax.set_title('Disease Prevalence in Evaluated IU X-Ray Subset',
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(diseases, rotation=35, ha='right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_summary(metrics_df, outpath):
    """Aggregate TP/FP/TN/FN heatmap across all diseases."""
    fig, axes = plt.subplots(2, 7, figsize=(18, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Confusion Matrix per Disease — RadVision AI',
                 fontsize=13, fontweight='bold', y=1.01)
    axes = axes.flatten()

    for i, row in metrics_df.iterrows():
        ax = axes[i]
        total = row['TP'] + row['FP'] + row['TN'] + row['FN']
        mat = np.array([[row['TP'], row['FN']],
                        [row['FP'], row['TN']]])
        im = ax.imshow(mat, cmap='Blues', vmin=0)
        ax.set_title(row['Disease'].replace('_', '\n'),
                     fontsize=8, fontweight='bold')
        for r in range(2):
            for c in range(2):
                pct = mat[r, c] / max(1, total) * 100
                ax.text(c, r, f'{int(mat[r,c])}\n({pct:.0f}%)',
                        ha='center', va='center', fontsize=7,
                        color='white' if mat[r,c] > total*0.4 else '#333')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Pred+','Pred−'], fontsize=7)
        ax.set_yticklabels(['GT+','GT−'], fontsize=7)
        ax.tick_params(length=0)

    for j in range(len(metrics_df), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_report_metrics(bleu_scores, rouge_scores, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('AI Report vs Real Radiologist Report — Text Similarity',
                 fontsize=13, fontweight='bold')

    bleu1 = [s['bleu1'] for s in bleu_scores]
    bleu2 = [s['bleu2'] for s in bleu_scores]
    rl    = rouge_scores

    # BLEU-1 histogram
    axes[0].hist(bleu1, bins=20, color='#3498db', edgecolor='white',
                 alpha=0.85, zorder=3)
    axes[0].axvline(np.mean(bleu1), color='#e74c3c', lw=2,
                    label=f'Mean={np.mean(bleu1):.3f}')
    axes[0].set_title('BLEU-1 Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('BLEU-1 Score'); axes[0].set_ylabel('Count')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle='--', zorder=0)

    # BLEU-2 histogram
    axes[1].hist(bleu2, bins=20, color='#2ecc71', edgecolor='white',
                 alpha=0.85, zorder=3)
    axes[1].axvline(np.mean(bleu2), color='#e74c3c', lw=2,
                    label=f'Mean={np.mean(bleu2):.3f}')
    axes[1].set_title('BLEU-2 Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('BLEU-2 Score'); axes[1].set_ylabel('Count')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--', zorder=0)

    # ROUGE-L histogram
    axes[2].hist(rl, bins=20, color='#9b59b6', edgecolor='white',
                 alpha=0.85, zorder=3)
    axes[2].axvline(np.mean(rl), color='#e74c3c', lw=2,
                    label=f'Mean={np.mean(rl):.3f}')
    axes[2].set_title('ROUGE-L Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('ROUGE-L Score'); axes[2].set_ylabel('Count')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3, linestyle='--', zorder=0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_distributions(all_preds, all_labels, outpath):
    """Probability distribution for positive vs negative cases."""
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    fig.patch.set_facecolor('white')
    fig.suptitle('Prediction Probability Distribution — Positive vs Negative Cases',
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for i, disease in enumerate(DISEASES):
        ax = axes[i]
        pos = [p[disease] for p, l in zip(all_preds, all_labels) if l[disease] == 1]
        neg = [p[disease] for p, l in zip(all_preds, all_labels) if l[disease] == 0]
        if pos:
            ax.hist(pos, bins=15, alpha=0.7, color='#e74c3c',
                    label=f'Positive (n={len(pos)})', density=True)
        if neg:
            ax.hist(neg, bins=15, alpha=0.7, color='#3498db',
                    label=f'Negative (n={len(neg)})', density=True)
        ax.set_title(disease.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Predicted prob.', fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    for j in range(len(DISEASES), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(metrics_df, n_images, mean_auc,
                           bleu1_mean, bleu2_mean, rouge_mean, outpath):
    metrics_df = metrics_df.dropna(subset=['AUC'])   # evaluable conditions only
    """One-page summary dashboard."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    title_style = dict(fontsize=11, fontweight='bold', color='white', pad=8)
    val_style   = dict(fontsize=28, fontweight='bold')

    # Top row — 3 metric cards
    kpi = [
        ('Images Evaluated', f'{n_images}',       '#4ecdc4'),
        ('Mean AUC',         f'{mean_auc:.3f}',   '#f5a623'),
        ('Mean BLEU-1',      f'{bleu1_mean:.3f}', '#c5b9f6'),
    ]
    for col, (label, value, color) in enumerate(kpi):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('#16213e')
        ax.text(0.5, 0.65, value, transform=ax.transAxes,
                ha='center', va='center', color=color, **val_style)
        ax.text(0.5, 0.25, label, transform=ax.transAxes,
                ha='center', va='center', color='#aaaaaa', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)

    # Bottom row — AUC bar
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor('#0f3460')
    diseases = metrics_df['Disease'].tolist()
    aucs     = metrics_df['AUC'].tolist()
    bar_colors = ['#4ecdc4' if a >= 0.85 else '#f5a623' if a >= 0.75 else '#e94560'
                  for a in aucs]
    bars = ax2.bar(diseases, aucs, color=bar_colors, edgecolor='#1a1a2e',
                   linewidth=0.5, zorder=3)
    ax2.axhline(np.mean(aucs), color='white', linestyle='--',
                lw=1.5, label=f'Mean = {np.mean(aucs):.3f}', zorder=4)
    for bar, a in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                 f'{a:.3f}', ha='center', va='bottom', fontsize=7.5,
                 color='white', fontweight='bold')
    ax2.set_ylim([0.3, 1.05])
    ax2.set_xticklabels(diseases, rotation=30, ha='right',
                        fontsize=9, color='white')
    ax2.set_ylabel('AUC', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.legend(fontsize=10, labelcolor='white',
               facecolor='#16213e', edgecolor='#555')
    ax2.set_title('Per-Disease AUC — RadVision AI × IU Chest X-Ray',
                  **title_style)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#555')

    fig.suptitle('RadVision AI — Evaluation Summary Dashboard',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    # Subtitle with ROUGE
    fig.text(0.5, 0.94,
             f'Dataset: Indiana University Chest X-Ray  |  '
             f'BLEU-2: {bleu2_mean:.3f}  |  ROUGE-L: {rouge_mean:.3f}  |  '
             f'Evaluated: {datetime.now().strftime("%d %b %Y %H:%M")}',
             ha='center', fontsize=9, color='#aaaaaa')

    plt.savefig(outpath, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(all_preds, all_labels, threshold=THRESHOLD):
    """Compute per-disease AUC, sensitivity, specificity, TP, FP, TN, FN."""
    from sklearn.metrics import roc_auc_score

    rows = []
    for disease in DISEASES:
        gt = np.array([l[disease] for l in all_labels])
        pr = np.array([p[disease] for p in all_preds])
        pos = gt.sum()

        if pos < 2 or (gt == 0).all():
            rows.append({
                'Disease': disease, 'AUC': np.nan,
                'Sensitivity': np.nan, 'Specificity': np.nan,
                'Accuracy': np.nan, 'Precision': np.nan,
                'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
                'N_positive': int(pos), 'N_total': len(gt)
            })
            continue

        try:
            auc_val = roc_auc_score(gt, pr)
        except Exception:
            auc_val = np.nan

        pred_bin = (pr >= threshold).astype(int)
        tp = int(((pred_bin == 1) & (gt == 1)).sum())
        fp = int(((pred_bin == 1) & (gt == 0)).sum())
        tn = int(((pred_bin == 0) & (gt == 0)).sum())
        fn = int(((pred_bin == 0) & (gt == 1)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy    = (tp + tn) / len(gt) if len(gt) > 0 else 0.0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        rows.append({
            'Disease':     disease,
            'AUC':         round(auc_val, 4),
            'Sensitivity': round(sensitivity, 4),
            'Specificity': round(specificity, 4),
            'Accuracy':    round(accuracy, 4),
            'Precision':   round(precision, 4),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'N_positive':  int(pos),
            'N_total':     len(gt)
        })

    df = pd.DataFrame(rows)
    mean_row = {
        'Disease':     'MEAN',
        'AUC':         round(df['AUC'].mean(), 4),
        'Sensitivity': round(df['Sensitivity'].mean(), 4),
        'Specificity': round(df['Specificity'].mean(), 4),
        'Accuracy':    round(df['Accuracy'].mean(), 4),
        'Precision':   round(df['Precision'].mean(), 4),
        'TP': df['TP'].sum(), 'FP': df['FP'].sum(),
        'TN': df['TN'].sum(), 'FN': df['FN'].sum(),
        'N_positive':  df['N_positive'].sum(),
        'N_total':     len(all_labels)
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="RadVision AI evaluation on IU X-Ray")
    p.add_argument("--location", choices=["gradcam", "prior", "none"], default="gradcam",
                   help="marker source given to the LLM (ablation switch)")
    p.add_argument("--tag", default=None, help="name of the output sub-folder")
    p.add_argument("--data-dir", default=None, help="IU X-Ray folder (images/, indiana_*.csv)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-images", type=int, default=MAX_IMAGES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-report", action="store_true", help="skip LLM report generation")
    return p.parse_args()


def main():
    global IMAGE_DIR, PROJ_CSV, REPORT_CSV, RUN_REPORT_EVAL
    args = parse_args()
    if args.data_dir:
        IMAGE_DIR  = os.path.join(args.data_dir, "images", "images_normalized")
        PROJ_CSV   = os.path.join(args.data_dir, "indiana_projections.csv")
        REPORT_CSV = os.path.join(args.data_dir, "indiana_reports.csv")
    if args.no_report:
        RUN_REPORT_EVAL = False
    use_gradcam = args.location == "gradcam"
    include_location = args.location != "none"
    tag = args.tag or f"location_{args.location}"
    out_dir = os.path.join(args.output_dir or OUTPUT_DIR, tag)

    print("=" * 60)
    print("  RadVision AI — Evaluation Pipeline")
    print("  Indiana University Chest X-Ray Dataset")
    print(f"  location={args.location}  seed={args.seed}  max_images={args.max_images}")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/6] Loading CSV files...")
    proj_df   = pd.read_csv(PROJ_CSV)
    report_df = pd.read_csv(REPORT_CSV)
    print(f"  Projections: {len(proj_df)} rows, {proj_df['uid'].nunique()} patients")
    print(f"  Reports:     {len(report_df)} rows")

    if FRONTAL_ONLY:
        proj_df = proj_df[proj_df['projection'] == 'Frontal'].copy()
        print(f"  Frontal views only: {len(proj_df)} images")

    merged = proj_df.merge(report_df[['uid', 'MeSH', 'findings', 'impression']],
                           on='uid', how='left')
    merged['full_path'] = merged['filename'].apply(lambda f: os.path.join(IMAGE_DIR, f))
    merged['exists'] = merged['full_path'].apply(os.path.exists)
    merged = merged[merged['exists']].copy()
    n_frontal_on_disk = len(merged)
    print(f"  Images found on disk: {n_frontal_on_disk}")
    if n_frontal_on_disk == 0:
        print("ERROR: No images found. Check --data-dir / IU_XRAY_DIR.")
        return

    # Simple random sample with a fixed seed (same images in every run)
    if args.max_images and len(merged) > args.max_images:
        merged = merged.sample(n=args.max_images, random_state=args.seed).reset_index(drop=True)
        print(f"  Sampled: {len(merged)} images (seed={args.seed})")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[2/6] Loading CheXNet model...")
    from xray_model import load_model, analyze_xray
    model, device = load_model()
    print(f"  Model output order: {list(model.pathologies)[:14]}")

    if RUN_REPORT_EVAL:
        from report import generate_report, MODEL_NAME
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY is not set (backend/.env). Use --no-report to skip reports.")
            return
        print(f"  Claude report generator loaded ({MODEL_NAME}, temperature={LLM_TEMPERATURE}).")

    # ── Run inference ────────────────────────────────────────────────────────
    print(f"\n[3/6] Running inference on {len(merged)} images...")
    all_preds, all_labels = [], []
    bleu_scores, rouge_scores = [], []
    results_rows = []
    n_report_errors = 0
    n_no_reference = 0

    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="Processing", unit="img"):
        img_path = row['full_path']
        gt_diseases = parse_mesh(row.get('MeSH', ''))
        gt_dict = {d: int(d in gt_diseases) for d in DISEASES}

        out = analyze_xray(img_path, model, device, use_gradcam=use_gradcam, return_probs=True)
        if out is None:
            print(f"\n  WARNING: Failed {row['filename']}")
            continue
        findings, pred_dict = out

        all_preds.append(pred_dict)
        all_labels.append(gt_dict)

        b1 = b2 = rl_score = np.nan
        ai_text, real_text, scored, report_error = '', '', False, False
        findings_txt = row.get('findings', '')
        impression_txt = row.get('impression', '')
        real_text = ' '.join(str(t) for t in (findings_txt, impression_txt) if pd.notna(t)).strip()

        if RUN_REPORT_EVAL:
            report = generate_report(findings, include_location=include_location,
                                     temperature=LLM_TEMPERATURE)
            report_error = bool(report.get('error', False))
            ai_text = (str(report.get('findings', '')) + ' ' +
                       str(report.get('impression', ''))).strip()
            if report_error:
                n_report_errors += 1
            elif not real_text:
                n_no_reference += 1
            elif ai_text:
                b1 = simple_bleu(real_text, ai_text, n=1)
                b2 = simple_bleu(real_text, ai_text, n=2)
                rl_score = rouge_l(real_text, ai_text)
                bleu_scores.append({'bleu1': b1, 'bleu2': b2})
                rouge_scores.append(rl_score)
                scored = True

        results_rows.append({
            'uid':          row['uid'],
            'filename':     row['filename'],
            'gt_diseases':  '|'.join(sorted(gt_diseases)) if gt_diseases else 'Normal',
            'top_pred':     findings[0]['name'] if findings else 'None',
            'top_prob':     findings[0]['probability'] if findings else 0,
            'n_findings':   len(findings),
            'n_gradcam_markers': sum(1 for f in findings[:5] if f.get('loc_source') == 'gradcam'),
            'scored':       scored,
            'report_error': report_error,
            'bleu1':        b1,
            'bleu2':        b2,
            'rouge_l':      rl_score,
            **{f'pred_{d}': pred_dict[d] for d in DISEASES},
            **{f'gt_{d}':   gt_dict[d] for d in DISEASES},
            'markers':      json.dumps([{k: f.get(k) for k in ('name', 'x', 'y', 'radius', 'loc_source')}
                                        for f in findings[:5]]),
            'ai_report':    ai_text,
            'reference_report': real_text,
        })

    n_eval = len(all_preds)
    print(f"\n  Successfully processed: {n_eval} images")
    if n_eval == 0:
        print("ERROR: No images were processed successfully.")
        return

    # ── Calculate metrics ────────────────────────────────────────────────────
    print("\n[4/6] Calculating metrics...")
    metrics_df = calculate_metrics(all_preds, all_labels)
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(metrics_df[['Disease', 'AUC', 'Sensitivity', 'Specificity', 'Precision',
                      'Accuracy', 'N_positive']].to_string(index=False))
    mean_auc = metrics_df[metrics_df['Disease'] != 'MEAN']['AUC'].mean()

    bleu1_mean = np.mean([s['bleu1'] for s in bleu_scores]) if bleu_scores else 0
    bleu2_mean = np.mean([s['bleu2'] for s in bleu_scores]) if bleu_scores else 0
    rouge_mean = np.mean(rouge_scores) if rouge_scores else 0
    if bleu_scores:
        print(f"\n  Report similarity vs real radiologist text (n={len(bleu_scores)} scored reports):")
        print(f"    BLEU-1:  {bleu1_mean:.4f}")
        print(f"    BLEU-2:  {bleu2_mean:.4f}")
        print(f"    ROUGE-L: {rouge_mean:.4f}")
        print(f"    Report errors: {n_report_errors}; images without reference text: {n_no_reference}")

    # ── Save results ─────────────────────────────────────────────────────────
    print("\n[5/6] Saving results...")
    pd.DataFrame(results_rows).to_csv(os.path.join(out_dir, f'per_image_results_{ts}.csv'), index=False)
    metrics_df.to_csv(os.path.join(out_dir, f'metrics_table_{ts}.csv'), index=False)
    try:
        import torchxrayvision as xrv
        xrv_version = getattr(xrv, '__version__', 'unknown')
    except Exception:
        xrv_version = 'unknown'
    with open(os.path.join(out_dir, f'run_config_{ts}.json'), 'w') as f:
        json.dump({
            'timestamp': ts, 'location_mode': args.location, 'use_gradcam': use_gradcam,
            'location_in_prompt': include_location, 'seed': args.seed,
            'sampling': 'simple random sample of frontal images (pandas.DataFrame.sample)',
            'max_images': args.max_images, 'frontal_images_on_disk': n_frontal_on_disk,
            'images_processed': n_eval, 'threshold': THRESHOLD,
            'model_weights': 'densenet121-res224-all', 'model_output_order': list(model.pathologies),
            'torchxrayvision_version': xrv_version,
            'llm': MODEL_NAME if RUN_REPORT_EVAL else None, 'llm_temperature': LLM_TEMPERATURE,
            'reports_scored': len(bleu_scores), 'report_errors': n_report_errors,
            'images_without_reference': n_no_reference,
            'mean_auc': float(mean_auc), 'bleu1': float(bleu1_mean), 'bleu2': float(bleu2_mean),
            'rouge_l': float(rouge_mean),
        }, f, indent=2)
    print(f"  Saved to: {out_dir}")

    # ── Generate plots ───────────────────────────────────────────────────────
    print("\n[6/6] Generating graphs...")
    plot_df = metrics_df[metrics_df['Disease'] != 'MEAN'].copy()
    plot_roc_curves(all_preds, all_labels, os.path.join(out_dir, f'roc_curves_{ts}.png'))
    plot_auc_bar(plot_df, os.path.join(out_dir, f'auc_per_disease_{ts}.png'))
    plot_sensitivity_specificity(plot_df, os.path.join(out_dir, f'sensitivity_specificity_{ts}.png'))
    plot_disease_prevalence(all_labels, os.path.join(out_dir, f'disease_prevalence_{ts}.png'))
    plot_confusion_summary(plot_df, os.path.join(out_dir, f'confusion_matrices_{ts}.png'))
    plot_prediction_distributions(all_preds, all_labels, os.path.join(out_dir, f'prediction_distributions_{ts}.png'))
    if bleu_scores:
        plot_report_metrics(bleu_scores, rouge_scores, os.path.join(out_dir, f'report_similarity_{ts}.png'))
    plot_summary_dashboard(plot_df, n_eval, mean_auc, bleu1_mean, bleu2_mean, rouge_mean,
                           os.path.join(out_dir, f'summary_dashboard_{ts}.png'))

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Images evaluated : {n_eval}")
    print(f"  Mean AUC         : {mean_auc:.4f}")
    print(f"  Output folder    : {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
