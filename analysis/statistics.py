"""
Statistics for the re-run RadVision AI evaluation (corrected label order).

Usage (from the repository root):
    python analysis/statistics.py <WITHOUT-GradCAM csv or run folder> <WITH-GradCAM csv or run folder>

Outputs (analysis/results_v2/):
    per_disease_metrics_with_CI.csv   AUC with stratified-bootstrap and DeLong 95% CIs,
                                      sensitivity, specificity, precision, accuracy
    ablation_wilcoxon.csv             paired Wilcoxon signed-rank tests, BLEU-1/BLEU-2/ROUGE-L
    STATISTICS_REPORT.md              everything above as tables for the paper
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_statistics import CODE_ORDER as DISEASES, bootstrap_auc_ci, delong_auc_ci, N_BOOT, SEED

THRESHOLD = 0.30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")


def metrics(df):
    rng = np.random.default_rng(SEED)
    rows = []
    for d in DISEASES:
        y = df[f"gt_{d}"].to_numpy().astype(int)
        s = df[f"pred_{d}"].to_numpy().astype(float)
        r = {"Disease": d, "N_pos": int(y.sum()), "N": len(y)}
        if y.sum() >= 2:
            p = (s >= THRESHOLD).astype(int)
            tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
            tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
            lo, hi = bootstrap_auc_ci(y, s, rng)
            _, dlo, dhi = delong_auc_ci(y, s)
            r.update(AUC=roc_auc_score(y, s), AUC_boot_lo=lo, AUC_boot_hi=hi, AUC_delong_lo=dlo,
                     AUC_delong_hi=dhi, Sensitivity=tp / (tp + fn), Specificity=tn / (tn + fp),
                     Precision=tp / (tp + fp) if tp + fp else 0.0, Accuracy=(tp + tn) / len(y),
                     TP=tp, FP=fp, TN=tn, FN=fn)
        rows.append(r)
    out = pd.DataFrame(rows)
    ev = out.dropna(subset=["AUC"])
    mean = {"Disease": f"MEAN ({len(ev)} evaluable)", "N_pos": int(ev.N_pos.sum())}
    for c in ["AUC", "Sensitivity", "Specificity", "Precision", "Accuracy"]:
        mean[c] = ev[c].mean()
    return pd.concat([out, pd.DataFrame([mean])], ignore_index=True)


def _latest(path):
    """Accept a CSV path or a run folder (then the newest per_image_results_*.csv is used)."""
    if os.path.isdir(path):
        import glob
        files = sorted(glob.glob(os.path.join(path, "per_image_results_*.csv")))
        if not files:
            sys.exit(f"No per_image_results_*.csv in {path}")
        return files[-1]
    return path


def main(path_without, path_with):
    path_without, path_with = _latest(path_without), _latest(path_with)
    os.makedirs(OUT, exist_ok=True)
    a, b = pd.read_csv(path_without), pd.read_csv(path_with)
    assert (a.filename.values == b.filename.values).all(), "the two runs used different images"
    pc = [f"pred_{d}" for d in DISEASES]
    L = []
    P = L.append
    P("# RadVision AI — re-run statistics (corrected label order)\n")
    P(f"- Run without GradCAM: `{os.path.basename(path_without)}`")
    P(f"- Run with GradCAM: `{os.path.basename(path_with)}`")
    P(f"- Images: {len(b)} (same images, same order in both runs)")
    P(f"- Largest classifier-score difference between runs: {float((a[pc] - b[pc]).abs().max().max()):.2e}")
    P(f"- Normal / abnormal: {(b.gt_diseases == 'Normal').sum()} / {(b.gt_diseases != 'Normal').sum()}")
    P(f"- Findings per image: mean {b.n_findings.mean():.2f}, SD {b.n_findings.std():.2f}, "
      f"range {b.n_findings.min()}–{b.n_findings.max()}")
    P(f"- Most frequent top prediction: {b.top_pred.value_counts().head(3).to_dict()}")
    if "n_gradcam_markers" in b:
        P(f"- GradCAM markers placed (with-GradCAM run): {int(b.n_gradcam_markers.sum())} "
          f"of {int(np.minimum(b.n_findings, 5).sum())} top-5 findings; prior fallback for the rest")

    m = metrics(b)
    m.round(4).to_csv(os.path.join(OUT, "per_disease_metrics_with_CI.csv"), index=False)
    P(f"\n## Classification (threshold {THRESHOLD}; {N_BOOT} stratified bootstrap resamples, seed {SEED})\n")
    P("| Disease | N+ | AUC | 95% CI (bootstrap) | 95% CI (DeLong) | Sens. | Spec. | Precision | Acc. |")
    P("|---|---|---|---|---|---|---|---|---|")
    for _, r in m.iterrows():
        if pd.isna(r.get("AUC")):
            P(f"| {r.Disease} | {int(r.N_pos)} | — | — | — | — | — | — | — |")
        elif str(r.Disease).startswith("MEAN"):
            P(f"| **{r.Disease}** | {int(r.N_pos)} | **{r.AUC:.3f}** | | | {r.Sensitivity:.3f} | "
              f"{r.Specificity:.3f} | {r.Precision:.3f} | {r.Accuracy:.3f} |")
        else:
            P(f"| {r.Disease} | {int(r.N_pos)} | {r.AUC:.3f} | {r.AUC_boot_lo:.3f}–{r.AUC_boot_hi:.3f} | "
              f"{r.AUC_delong_lo:.3f}–{r.AUC_delong_hi:.3f} | {r.Sensitivity:.3f} | {r.Specificity:.3f} | "
              f"{r.Precision:.3f} | {r.Accuracy:.3f} |")

    both = (a.scored.astype(bool) & b.scored.astype(bool)).to_numpy()
    P(f"\n## Report generation and GradCAM ablation (paired over {both.sum()} images scored in both runs)\n")
    P(f"Reports scored: {int(a.scored.sum())} without / {int(b.scored.sum())} with GradCAM; "
      f"API errors: {int(a.report_error.sum())} / {int(b.report_error.sum())}.\n")
    rng = np.random.default_rng(SEED)
    rows = []
    for mt in ["bleu1", "bleu2", "rouge_l"]:
        x, y = a[mt].to_numpy()[both], b[mt].to_numpy()[both]
        d = y - x
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
        nonzero = d[d != 0]
        p = stats.wilcoxon(y, x).pvalue if len(nonzero) else 1.0
        rows.append(dict(Metric=mt, n_pairs=int(both.sum()), mean_without=x.mean(), sd_without=x.std(ddof=1),
                         mean_with=y.mean(), sd_with=y.std(ddof=1), mean_diff=d.mean(),
                         diff_CI_lo=np.percentile(boots, 2.5), diff_CI_hi=np.percentile(boots, 97.5),
                         n_better=int((d > 0).sum()), n_worse=int((d < 0).sum()), n_tied=int((d == 0).sum()),
                         wilcoxon_p=p))
    w = pd.DataFrame(rows)
    w.round(4).to_csv(os.path.join(OUT, "ablation_wilcoxon.csv"), index=False)
    P("| Metric | Without GradCAM (mean ± SD) | With GradCAM (mean ± SD) | Δ (95% CI) | Better / worse / tied | Wilcoxon p |")
    P("|---|---|---|---|---|---|")
    for _, r in w.iterrows():
        P(f"| {r.Metric} | {r.mean_without:.4f} ± {r.sd_without:.4f} | {r.mean_with:.4f} ± {r.sd_with:.4f} | "
          f"{r.mean_diff:+.4f} ({r.diff_CI_lo:+.4f} to {r.diff_CI_hi:+.4f}) | {r.n_better} / {r.n_worse} / {r.n_tied} | "
          f"{r.wilcoxon_p:.3f} |")

    with open(os.path.join(OUT, "STATISTICS_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
