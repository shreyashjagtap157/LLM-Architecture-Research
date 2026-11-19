# Source-identified v9 pilot repeat — 2026-11-19

**Run timestamp (UTC):** 2026-11-19T07:52:46Z  
**Run class:** exploratory CPU character-LM feasibility repeat  
**Profiles:** P0, P1, P1-LA  
**Seeds:** 20261118, 20261119  
**Steps:** 500 optimizer steps per profile/seed (six complete runs)  
**Data:** bundled 29,926-byte Tiny Shakespeare excerpt; 23,940 / 2,993 / 2,993 split  
**Device:** CPU; PyTorch 2.10.0+cpu

## Exact run identity

- `train_pilot.py`: `64bbc094f1773c31307dce0215c182bc06bcfe6b051391169d34885bf7489f4a`
- `prepare_hf_dataset.py`: `c119967fb5548fc24478c14fade3d28b41aa38e4023ddb1141047105f41a4d00`
- `results/dataset_manifest.json`: `f575acf98eeb16196f78521061849a93def228b9b582366f78980bd0ad8ea597`

All six run records and their checkpoint SHA-256 values are in `results/pilot_results.json`. The report states all 500 optimizer steps completed and all losses/gradients remained finite.

## Mean results

| Profile | Parameters | Mean test NLL (nats/char) | Sample SD | Mean top-1 accuracy | Mean training seconds |
|---|---:|---:|---:|---:|---:|
| P0 dense attention | 72,832 | 2.7708028 | 0.0443666 | 25.6216% | 1.8691 |
| P1 GDN-2-inspired + attention | 72,928 | 2.7411011 | 0.0322888 | 26.8817% | 7.1623 |
| P1-LA local/global attention | 72,832 | 2.7597540 | 0.0451104 | 25.8905% | 2.0322 |
| Smoothed unigram baseline | — | 3.2554766 | — | 15.1209% | — |
| Smoothed bigram baseline | — | **2.4317889** | — | **27.4530%** | — |

P1 beats P0 modestly but takes about 3.83× the mean training time in this sequential CPU reference, and all neural candidates remain worse than the bigram baseline on this corpus. No architecture promotion is justified.

## Reproduction interpretation

This is a new source-identified repeat of the v9 pilot. The aggregate loss/accuracy values match the earlier recorded v9 aggregate; training-time measurements differ slightly due runtime conditions. This does not retroactively identify the exact source bytes of older runs, nor does it validate v10/V11 dynamic computation.

## Operational warning

`train_pilot.py` is an unconditional experiment entry point. It does not parse `--help`; invoking the script starts the full six-run experiment and overwrites result files in `results/`. The `--help` invocation during packaging therefore started this run; it completed successfully and its source identities are recorded above. Do not use it for a harmless CLI smoke test.
