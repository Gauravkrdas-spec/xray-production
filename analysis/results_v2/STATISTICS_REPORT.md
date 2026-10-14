# RadVision AI — re-run statistics (corrected label order)

- Run without GradCAM: `per_image_results_20261014_000006.csv`
- Run with GradCAM: `per_image_results_20261014_002413.csv`
- Images: 200 (same images, same order in both runs)
- Largest classifier-score difference between runs: 0.00e+00
- Normal / abnormal: 140 / 60
- Findings per image: mean 7.93, SD 3.22, range 1–14
- Most frequent top prediction: {'Fibrosis': 51, 'Infiltration': 30, 'Cardiomegaly': 26}
- GradCAM markers placed (with-GradCAM run): 935 of 935 top-5 findings; prior fallback for the rest

## Classification (threshold 0.3; 2000 stratified bootstrap resamples, seed 2026)

| Disease | N+ | AUC | 95% CI (bootstrap) | 95% CI (DeLong) | Sens. | Spec. | Precision | Acc. |
|---|---|---|---|---|---|---|---|---|
| Atelectasis | 11 | 0.754 | 0.641–0.856 | 0.645–0.863 | 0.636 | 0.656 | 0.097 | 0.655 |
| Cardiomegaly | 21 | 0.921 | 0.861–0.968 | 0.867–0.974 | 0.905 | 0.799 | 0.345 | 0.810 |
| Effusion | 6 | 0.862 | 0.621–0.992 | 0.634–1.000 | 0.833 | 0.845 | 0.143 | 0.845 |
| Infiltration | 5 | 0.773 | 0.498–0.979 | 0.483–1.000 | 0.800 | 0.497 | 0.039 | 0.505 |
| Mass | 12 | 0.739 | 0.567–0.881 | 0.581–0.897 | 0.750 | 0.574 | 0.101 | 0.585 |
| Nodule | 3 | 0.665 | 0.107–0.983 | 0.111–1.000 | 0.667 | 0.472 | 0.019 | 0.475 |
| Pneumonia | 0 | — | — | — | — | — | — | — |
| Pneumothorax | 0 | — | — | — | — | — | — | — |
| Consolidation | 0 | — | — | — | — | — | — | — |
| Edema | 0 | — | — | — | — | — | — | — |
| Emphysema | 3 | 0.755 | 0.609–0.909 | 0.576–0.933 | 1.000 | 0.604 | 0.037 | 0.610 |
| Fibrosis | 0 | — | — | — | — | — | — | — |
| Pleural_Thickening | 0 | — | — | — | — | — | — | — |
| Hernia | 2 | 0.982 | 0.944–1.000 | 0.945–1.000 | 1.000 | 0.939 | 0.143 | 0.940 |
| **MEAN (8 evaluable)** | 63 | **0.806** | | | 0.824 | 0.673 | 0.116 | 0.678 |

## Report generation and GradCAM ablation (paired over 197 images scored in both runs)

Reports scored: 197 without / 197 with GradCAM; API errors: 0 / 0.

| Metric | Without GradCAM (mean ± SD) | With GradCAM (mean ± SD) | Δ (95% CI) | Better / worse / tied | Wilcoxon p |
|---|---|---|---|---|---|
| bleu1 | 0.0905 ± 0.0523 | 0.0908 ± 0.0540 | +0.0003 (-0.0040 to +0.0043) | 93 / 98 / 6 | 0.883 |
| bleu2 | 0.0171 ± 0.0272 | 0.0175 ± 0.0297 | +0.0004 (-0.0021 to +0.0029) | 39 / 34 / 124 | 0.662 |
| rouge_l | 0.0852 ± 0.0406 | 0.0853 ± 0.0395 | +0.0001 (-0.0041 to +0.0045) | 103 / 91 / 3 | 0.812 |
