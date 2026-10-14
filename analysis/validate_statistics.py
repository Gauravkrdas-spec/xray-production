"""
Statistical validation of the RadVision AI IU X-Ray evaluation.

Uses ONLY the per-image outputs saved by backend/evaluation_pipeline.py:
  evaluation_results/per_image_results_20260521_215006.csv              (run labelled "with GradCAM")
  evaluation_results_without_radcam/per_image_results_20260521_205111.csv (run labelled "without GradCAM")

What it does
  1. Reproduces the published per-disease metrics (AUC, sensitivity, specificity, accuracy)
     and adds precision (PPV) at the 0.30 threshold.
  2. 95% confidence intervals for every AUC: stratified bootstrap (2,000 resamples) and DeLong.
  3. Paired Wilcoxon signed-rank tests for the BLEU-1 / BLEU-2 / ROUGE-L ablation difference.
  4. Re-analysis with the CORRECT torchxrayvision output order. backend/xray_model.py maps output
     index i to the NIH-ordered DISEASES[i], but densenet121-res224-all emits its outputs in
     torchxrayvision's own order, so the saved column pred_<X> actually holds the score of
     XRV_ORDER[i]. Because all 14 outputs were saved, the true score of every disease can be
     recovered from the CSV without re-running the model.

Run:  python analysis/validate_statistics.py      (numpy, pandas, scipy, scikit-learn)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_WITH = os.path.join(ROOT, "evaluation_results", "per_image_results_20260521_215006.csv")
RUN_WITHOUT = os.path.join(ROOT, "evaluation_results_without_radcam", "per_image_results_20260521_205111.csv")
OUT = os.path.join(ROOT, "analysis", "results")
os.makedirs(OUT, exist_ok=True)

THRESHOLD = 0.30
N_BOOT = 2000
SEED = 2026

# Order used in backend/xray_model.py (NIH ChestX-ray14 order)
CODE_ORDER = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
              'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
              'Pleural_Thickening', 'Hernia']
# Actual output order of torchxrayvision DenseNet(weights="densenet121-res224-all")
# (torchxrayvision/models.py, model_urls['all']['labels'], first 14 entries)
XRV_ORDER = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
             'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
             'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']


# ---------------------------------------------------------------- DeLong (Sun & Xu, 2014)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_auc_ci(y, s, alpha=0.95):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.concatenate([pos, neg]))
    auc = tz[:m].sum() / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    var = (np.var(v01, ddof=1) if m > 1 else 0.0) / m + np.var(v10, ddof=1) / n
    se = np.sqrt(var)
    zc = stats.norm.ppf(1 - (1 - alpha) / 2)
    return auc, max(0.0, auc - zc * se), min(1.0, auc + zc * se)


def bootstrap_auc_ci(y, s, rng, n_boot=N_BOOT):
    """Stratified bootstrap: positives and negatives resampled separately (keeps N+ fixed)."""
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
        aucs[b] = roc_auc_score(y[idx], s[idx])
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def per_disease(df, score_col_for, label):
    rng = np.random.default_rng(SEED)
    rows = []
    for d in CODE_ORDER:
        y = df[f"gt_{d}"].to_numpy().astype(int)
        s = df[score_col_for[d]].to_numpy().astype(float)
        npos = int(y.sum())
        r = {"Analysis": label, "Disease": d, "Score_column": score_col_for[d], "N_pos": npos, "N": len(y)}
        if npos < 2:
            rows.append(r)
            continue
        pred = (s >= THRESHOLD).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        auc = roc_auc_score(y, s)
        b_lo, b_hi = bootstrap_auc_ci(y, s, rng)
        _, d_lo, d_hi = delong_auc_ci(y, s)
        r.update(AUC=auc, AUC_boot_lo=b_lo, AUC_boot_hi=b_hi, AUC_delong_lo=d_lo, AUC_delong_hi=d_hi,
                 Sensitivity=tp / (tp + fn), Specificity=tn / (tn + fp),
                 Precision=(tp / (tp + fp)) if (tp + fp) else 0.0,
                 Accuracy=(tp + tn) / len(y), TP=tp, FP=fp, TN=tn, FN=fn)
        rows.append(r)
    out = pd.DataFrame(rows)
    ev = out.dropna(subset=["AUC"])
    mean = {"Analysis": label, "Disease": f"MEAN ({len(ev)} evaluable)"}
    for c in ["AUC", "Sensitivity", "Specificity", "Precision", "Accuracy"]:
        mean[c] = ev[c].mean()
    mean["N_pos"] = int(ev["N_pos"].sum())
    return pd.concat([out, pd.DataFrame([mean])], ignore_index=True)


def main():
    w = pd.read_csv(RUN_WITH)
    wo = pd.read_csv(RUN_WITHOUT)
    assert (w["filename"].values == wo["filename"].values).all(), "runs are not on the same images"
    lines = []
    P = lines.append

    # ------------------------------------------------------------ sample checks
    pc = [c for c in w.columns if c.startswith("pred_")]
    P("# Statistical validation of the RadVision AI IU X-Ray evaluation\n")
    P("## 0. Sample checks (from the saved per-image outputs)\n")
    P(f"- Images: {len(w)} in each run, same images in the same order: True")
    P(f"- Largest difference in any disease score between the two runs: {float((w[pc]-wo[pc]).abs().max().max())}")
    P(f"- Normal (no mapped MeSH label): {(w.gt_diseases=='Normal').sum()}; abnormal: {(w.gt_diseases!='Normal').sum()}")
    P(f"- Findings per image: mean {w.n_findings.mean():.3f}, SD {w.n_findings.std():.3f}, range {w.n_findings.min()}–{w.n_findings.max()}")
    P(f"- Most frequent top prediction: {w.top_pred.value_counts().index[0]} ({w.top_pred.value_counts().iloc[0]} images)")

    # ------------------------------------------------------------ 1+2 as published
    as_pub = per_disease(w, {d: f"pred_{d}" for d in CODE_ORDER}, "As published (code label order)")
    # ------------------------------------------------------------ 4 corrected
    corrected_col = {d: f"pred_{CODE_ORDER[XRV_ORDER.index(d)]}" for d in CODE_ORDER}
    corr = per_disease(w, corrected_col, "Corrected (torchxrayvision output order)")
    both = pd.concat([as_pub, corr], ignore_index=True)
    both.round(4).to_csv(os.path.join(OUT, "per_disease_metrics_with_CI.csv"), index=False)

    def table(df):
        t = ["| Disease | N+ | AUC | 95% CI (bootstrap) | 95% CI (DeLong) | Sens. | Spec. | Precision | Acc. |",
             "|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            if pd.isna(r.get("AUC")):
                t.append(f"| {r.Disease} | {int(r.N_pos)} | — | — | — | — | — | — | — |"); continue
            if str(r.Disease).startswith("MEAN"):
                t.append(f"| **{r.Disease}** | {int(r.N_pos)} | **{r.AUC:.3f}** | | | {r.Sensitivity:.3f} | {r.Specificity:.3f} | {r.Precision:.3f} | {r.Accuracy:.3f} |"); continue
            t.append(f"| {r.Disease} | {int(r.N_pos)} | {r.AUC:.3f} | {r.AUC_boot_lo:.3f}–{r.AUC_boot_hi:.3f} | "
                     f"{r.AUC_delong_lo:.3f}–{r.AUC_delong_hi:.3f} | {r.Sensitivity:.3f} | {r.Specificity:.3f} | {r.Precision:.3f} | {r.Accuracy:.3f} |")
        return "\n".join(t)

    P("\n## 1–2. Per-disease metrics as published, with precision and 95% CIs\n")
    P(f"Threshold {THRESHOLD}; stratified bootstrap with {N_BOOT} resamples (seed {SEED}); DeLong asymptotic CI.\n")
    P(table(as_pub))

    # ------------------------------------------------------------ 3 Wilcoxon
    P("\n## 3. Ablation: paired Wilcoxon signed-rank tests (with vs without GradCAM run)\n")
    zero_w = (w.bleu1 == 0) & (w.bleu2 == 0) & (w.rouge_l == 0)
    zero_wo = (wo.bleu1 == 0) & (wo.bleu2 == 0) & (wo.rouge_l == 0)
    rng = np.random.default_rng(SEED)
    wrows = []
    for subset, mask in [("All 200 images (as averaged in Table 4)", np.ones(len(w), bool)),
                         ("Images scored in both runs", ~(zero_w | zero_wo).to_numpy())]:
        for m in ["bleu1", "bleu2", "rouge_l"]:
            a, b = w[m].to_numpy()[mask], wo[m].to_numpy()[mask]
            diff = a - b
            res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
            wrows.append({"Subset": subset, "Metric": m, "n_pairs": int(mask.sum()),
                          "mean_with": a.mean(), "mean_without": b.mean(), "mean_diff": diff.mean(),
                          "diff_CI_lo": np.percentile(boots, 2.5), "diff_CI_hi": np.percentile(boots, 97.5),
                          "n_improved": int((diff > 0).sum()), "n_worse": int((diff < 0).sum()),
                          "n_tied": int((diff == 0).sum()), "W": res.statistic, "p_value": res.pvalue})
    wt = pd.DataFrame(wrows)
    wt.round(4).to_csv(os.path.join(OUT, "ablation_wilcoxon.csv"), index=False)
    P("| Subset | Metric | Pairs | Mean with | Mean without | Δ (95% CI) | Better / worse / tied | Wilcoxon p |")
    P("|---|---|---|---|---|---|---|---|")
    for _, r in wt.iterrows():
        P(f"| {r.Subset} | {r.Metric} | {r.n_pairs} | {r.mean_with:.4f} | {r.mean_without:.4f} | "
          f"{r.mean_diff:+.4f} ({r.diff_CI_lo:+.4f} to {r.diff_CI_hi:+.4f}) | {r.n_improved} / {r.n_worse} / {r.n_tied} | {r.p_value:.3f} |")
    same_txt = int(((w.bleu1 == wo.bleu1) & (w.bleu2 == wo.bleu2) & (w.rouge_l == wo.rouge_l)).sum())
    P(f"\n- Images with identical disease scores in both runs: {len(w)} of {len(w)}.")
    P(f"- Images whose text scores were nevertheless identical in both runs: {same_txt} of {len(w)}.")
    P(f"- Images scored 0 on all three text metrics: {int(zero_w.sum())} (with) and {int(zero_wo.sum())} (without).")
    P(f"- Images where BLEU-2 equals BLEU-1 (no bigram overlap; see code note): {int((w.bleu1 == w.bleu2).sum())} of {len(w)}.")

    P("\n## 4. Re-analysis with the correct torchxrayvision output order\n")
    P("| Disease | Score actually stored in the column the code labels as this disease |\n|---|---|")
    for i, d in enumerate(CODE_ORDER):
        P(f"| {d} | {XRV_ORDER[i]}{' (correct)' if XRV_ORDER[i] == d else ''} |")
    P("\nMetrics with each disease scored by its true output:\n")
    P(table(corr))

    # top-prediction distribution under the correct mapping
    true_scores = pd.DataFrame({d: w[corrected_col[d]] for d in CODE_ORDER})
    P(f"\n- Most frequent true top prediction: {true_scores.idxmax(axis=1).value_counts().head(3).to_dict()}")

    with open(os.path.join(OUT, "VALIDATION_REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
