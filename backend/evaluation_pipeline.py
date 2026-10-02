"""
RadVision AI — Evaluation Pipeline
Indiana University Chest X-Ray Dataset (IU X-Ray)

Run from your backend folder:
    python evaluation_pipeline.py

Results saved to OUTPUT_DIR (see config below).
"""

import os, sys, json, time, warnings
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
BACKEND_DIR = r"C:\Users\gaura\xray-production\backend"
sys.path.insert(0, BACKEND_DIR)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these paths if needed
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_DIR   = r"C:\Users\gaura\OneDrive\Desktop\xray_project\iu_xray\images\images_normalized"
PROJ_CSV    = r"C:\Users\gaura\OneDrive\Desktop\xray_project\iu_xray\indiana_projections.csv"
REPORT_CSV  = r"C:\Users\gaura\OneDrive\Desktop\xray_project\iu_xray\indiana_reports.csv"
OUTPUT_DIR  = r"C:\Users\gaura\OneDrive\Desktop\xray_project\evaluation_results"

# How many images to evaluate
# 200  → ~15 minutes, good for testing
# 500  → ~40 minutes, good for paper
# None → full dataset (~3818 images, ~5 hours)
MAX_IMAGES = 200

# Set True to also generate Claude reports and compute BLEU/ROUGE
# Each call costs ~$0.01 — 200 images ≈ $2.00
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

def simple_bleu(reference, hypothesis, n=2):
    """Compute BLEU-n between two text strings without NLTK."""
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    ref_tokens = str(reference).lower().split()
    hyp_tokens = str(hypothesis).lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    scores = []
    for i in range(1, n+1):
        ref_ng = ngrams(ref_tokens, i)
        hyp_ng = ngrams(hyp_tokens, i)
        if not hyp_ng:
            scores.append(0.0)
            continue
        ref_count = {}
        for ng in ref_ng:
            ref_count[ng] = ref_count.get(ng, 0) + 1
        matches = 0
        for ng in hyp_ng:
            if ref_count.get(ng, 0) > 0:
                matches += 1
                ref_count[ng] -= 1
        scores.append(matches / len(hyp_ng))

    if not scores or all(s == 0 for s in scores):
        return 0.0

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / max(1, len(ref_tokens)))
    geo_mean = np.exp(np.mean([np.log(s) for s in scores if s > 0])) if any(s > 0 for s in scores) else 0
    return round(bp * geo_mean, 4)


def rouge_l(reference, hypothesis):
    """Compute ROUGE-L F1 (longest common subsequence)."""
    ref  = str(reference).lower().split()
    hyp  = str(hypothesis).lower().split()
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
    return round(2 * prec * rec / (prec + rec), 4)


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

    diseases = metrics_df['Disease'].tolist()
    aucs     = metrics_df['AUC'].tolist()
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
    ax.legend(handles=legend_handles, fontsize=9, loc='upper right')

    ax.set_xlabel('Pathology', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Per-Disease AUC — RadVision AI on IU Chest X-Ray Dataset',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0.4, 1.05])
    ax.set_xticklabels(diseases, rotation=35, ha='right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sensitivity_specificity(metrics_df, outpath):
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

def calculate_metrics(all_preds, all_labels, threshold=0.3):
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
                'Accuracy': np.nan,
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

        rows.append({
            'Disease':     disease,
            'AUC':         round(auc_val, 4),
            'Sensitivity': round(sensitivity, 4),
            'Specificity': round(specificity, 4),
            'Accuracy':    round(accuracy, 4),
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

def main():
    print("=" * 60)
    print("  RadVision AI — Evaluation Pipeline")
    print("  Indiana University Chest X-Ray Dataset")
    print("=" * 60)

    # ── Create output directory ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/6] Loading CSV files...")
    proj_df   = pd.read_csv(PROJ_CSV)
    report_df = pd.read_csv(REPORT_CSV)

    print(f"  Projections: {len(proj_df)} rows, {proj_df['uid'].nunique()} patients")
    print(f"  Reports:     {len(report_df)} rows")

    # ── Filter frontal views ─────────────────────────────────────────────────
    if FRONTAL_ONLY:
        proj_df = proj_df[proj_df['projection'] == 'Frontal'].copy()
        print(f"  Frontal views only: {len(proj_df)} images")

    # ── Merge with report labels ─────────────────────────────────────────────
    merged = proj_df.merge(report_df[['uid','MeSH','findings','impression']],
                           on='uid', how='left')

    # ── Keep only images that exist on disk ──────────────────────────────────
    merged['full_path'] = merged['filename'].apply(
        lambda f: os.path.join(IMAGE_DIR, f))
    merged['exists'] = merged['full_path'].apply(os.path.exists)
    merged = merged[merged['exists']].copy()
    print(f"  Images found on disk: {len(merged)}")

    if len(merged) == 0:
        print("ERROR: No images found. Check IMAGE_DIR path.")
        return

    # ── Limit sample size ────────────────────────────────────────────────────
    if MAX_IMAGES and len(merged) > MAX_IMAGES:
        merged = merged.sample(n=MAX_IMAGES, random_state=42).reset_index(drop=True)
        print(f"  Sampled: {len(merged)} images (MAX_IMAGES={MAX_IMAGES})")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[2/6] Loading CheXNet model...")
    try:
        from xray_model import load_model, analyze_xray
        model, device = load_model()
    except ImportError:
        print("ERROR: Cannot import xray_model. Make sure BACKEND_DIR is correct.")
        print(f"  BACKEND_DIR = {BACKEND_DIR}")
        return

    if RUN_REPORT_EVAL:
        try:
            from report import generate_report
            print("  Claude report generator loaded.")
        except ImportError:
            print("  WARNING: Cannot import report.py — skipping report evaluation.")
            globals()['RUN_REPORT_EVAL'] = False

    # ── Run inference ────────────────────────────────────────────────────────
    print(f"\n[3/6] Running inference on {len(merged)} images...")
    print("  (This will take a while — grab a coffee!)")

    all_preds    = []   # list of dicts {disease: probability}
    all_labels   = []   # list of dicts {disease: 0 or 1}
    bleu_scores  = []   # list of dicts {bleu1, bleu2}
    rouge_scores = []   # list of floats

    results_rows = []

    for idx, row in tqdm(merged.iterrows(), total=len(merged),
                         desc="Processing", unit="img"):
        img_path = row['full_path']

        # Ground truth from MeSH
        gt_diseases = parse_mesh(row.get('MeSH', ''))
        gt_dict = {d: int(d in gt_diseases) for d in DISEASES}

        # CheXNet inference
        try:
            findings = analyze_xray(img_path, model, device)
        except Exception as e:
            print(f"\n  WARNING: Failed {row['filename']}: {e}")
            continue

        # Prediction probabilities dict
        pred_dict = {d: 0.0 for d in DISEASES}
        for f in findings:
            if f['name'] in pred_dict:
                pred_dict[f['name']] = f['probability'] / 100.0

        all_preds.append(pred_dict)
        all_labels.append(gt_dict)

        # Report quality evaluation
        b1, b2, rl_score = 0.0, 0.0, 0.0
        ai_report_text = ''

        if RUN_REPORT_EVAL:
            try:
                report = generate_report(findings)
                ai_text = (report.get('findings', '') + ' ' +
                           report.get('impression', '')).strip()
                real_text = (str(row.get('findings', '')) + ' ' +
                             str(row.get('impression', ''))).strip()

                if ai_text and real_text and real_text != 'nan nan':
                    b1 = simple_bleu(real_text, ai_text, n=1)
                    b2 = simple_bleu(real_text, ai_text, n=2)
                    rl_score = rouge_l(real_text, ai_text)
                    bleu_scores.append({'bleu1': b1, 'bleu2': b2})
                    rouge_scores.append(rl_score)
                    ai_report_text = ai_text[:200]
            except Exception as e:
                pass

        # Save per-image result
        results_rows.append({
            'uid':          row['uid'],
            'filename':     row['filename'],
            'gt_diseases':  '|'.join(gt_diseases) if gt_diseases else 'Normal',
            'top_pred':     findings[0]['name'] if findings else 'None',
            'top_prob':     findings[0]['probability'] if findings else 0,
            'n_findings':   len(findings),
            'bleu1':        round(b1, 4),
            'bleu2':        round(b2, 4),
            'rouge_l':      round(rl_score, 4),
            **{f'pred_{d}': round(pred_dict[d], 4) for d in DISEASES},
            **{f'gt_{d}':   gt_dict[d] for d in DISEASES},
        })

    n_eval = len(all_preds)
    print(f"\n  Successfully processed: {n_eval} images")

    if n_eval == 0:
        print("ERROR: No images were processed successfully.")
        return

    # ── Calculate metrics ────────────────────────────────────────────────────
    print("\n[4/6] Calculating metrics...")
    metrics_df = calculate_metrics(all_preds, all_labels)

    # Print to console
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    display_cols = ['Disease', 'AUC', 'Sensitivity', 'Specificity',
                    'Accuracy', 'N_positive']
    print(metrics_df[display_cols].to_string(index=False))

    mean_auc = metrics_df[metrics_df['Disease'] != 'MEAN']['AUC'].mean()

    bleu1_mean = np.mean([s['bleu1'] for s in bleu_scores]) if bleu_scores else 0
    bleu2_mean = np.mean([s['bleu2'] for s in bleu_scores]) if bleu_scores else 0
    rouge_mean = np.mean(rouge_scores) if rouge_scores else 0

    if bleu_scores:
        print(f"\n  Report Similarity vs Real Radiologist Text:")
        print(f"    BLEU-1:  {bleu1_mean:.4f}")
        print(f"    BLEU-2:  {bleu2_mean:.4f}")
        print(f"    ROUGE-L: {rouge_mean:.4f}")

    # ── Save CSV results ─────────────────────────────────────────────────────
    print("\n[5/6] Saving results...")

    # Per-image results
    results_csv = os.path.join(OUTPUT_DIR, f'per_image_results_{ts}.csv')
    pd.DataFrame(results_rows).to_csv(results_csv, index=False)
    print(f"  Saved: per_image_results_{ts}.csv")

    # Metrics table
    metrics_csv = os.path.join(OUTPUT_DIR, f'metrics_table_{ts}.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  Saved: metrics_table_{ts}.csv")

    # ── Generate plots ───────────────────────────────────────────────────────
    print("\n[6/6] Generating graphs...")

    # Only use non-MEAN rows for plots
    plot_df = metrics_df[metrics_df['Disease'] != 'MEAN'].copy()

    plot_roc_curves(all_preds, all_labels,
        os.path.join(OUTPUT_DIR, f'roc_curves_{ts}.png'))
    print("  Saved: roc_curves.png")

    plot_auc_bar(plot_df,
        os.path.join(OUTPUT_DIR, f'auc_per_disease_{ts}.png'))
    print("  Saved: auc_per_disease.png")

    plot_sensitivity_specificity(plot_df,
        os.path.join(OUTPUT_DIR, f'sensitivity_specificity_{ts}.png'))
    print("  Saved: sensitivity_specificity.png")

    plot_disease_prevalence(all_labels,
        os.path.join(OUTPUT_DIR, f'disease_prevalence_{ts}.png'))
    print("  Saved: disease_prevalence.png")

    plot_confusion_summary(plot_df,
        os.path.join(OUTPUT_DIR, f'confusion_matrices_{ts}.png'))
    print("  Saved: confusion_matrices.png")

    plot_prediction_distributions(all_preds, all_labels,
        os.path.join(OUTPUT_DIR, f'prediction_distributions_{ts}.png'))
    print("  Saved: prediction_distributions.png")

    if bleu_scores:
        plot_report_metrics(bleu_scores, rouge_scores,
            os.path.join(OUTPUT_DIR, f'report_similarity_{ts}.png'))
        print("  Saved: report_similarity.png")

    plot_summary_dashboard(
        plot_df, n_eval, mean_auc,
        bleu1_mean, bleu2_mean, rouge_mean,
        os.path.join(OUTPUT_DIR, f'summary_dashboard_{ts}.png'))
    print("  Saved: summary_dashboard.png")

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EVALUATION COMPLETE")
    print("="*60)
    print(f"  Images evaluated : {n_eval}")
    print(f"  Mean AUC         : {mean_auc:.4f}")
    print(f"  Output folder    : {OUTPUT_DIR}")
    print(f"\n  Files saved:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if ts in f:
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"    {f}  ({size//1024} KB)")
    print("\n  Use these real results in your paper! They are 100% measured.")
    print("="*60)


if __name__ == '__main__':
    main()