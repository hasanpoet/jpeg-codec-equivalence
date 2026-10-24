# Verification record

Verified on Windows x86-64 with Python 3.11.9, using a fresh virtual environment and only the extracted code. Original dataset downloads were supplied as read-only inputs. No dataset images, virtual environment, or local verification workspace are included in the commit.

| Check | Result |
|---|---|
| Install pinned requirements | Passed; `pip check` reports no broken requirements |
| Unit tests | 25 passed |
| Regenerate without images or encoding | Passed in 142.766 seconds; 109 values |
| Values and three table bodies | Byte-identical to the source paper artifacts |
| Three figures | Rendered pixels identical; PDF metadata can differ |
| Five images per domain | All 20 reconstructed crop hashes match |
| End-to-end smoke test | Passed: 180 pairs, 720 search targets, PSNR/SSIM models, headers, reference robustness, payload analysis, and paper generation |
| Initial 612-pair replay | Zero scientific or classification differences; 916.75 seconds |
| Encoder-configuration replay | All 60 per-image rows and 12 aggregate rows match; elapsed-time fields excluded |
| Public-file privacy scan | No personal paths, email addresses, credential patterns, or attribution boilerplate found |
| Source preservation | Snapshot hashes unchanged and original tracked worktree clean |

The smoke test used an explicit group-disjoint split with three training and two test images per domain (12 training groups and eight test groups). It produced 840 PSNR model-evaluation rows, 278 SSIM model-evaluation rows, 880 header records and 16 reference-fit rows. Small-sample estimates are not compared with published full-sample estimates.

The 612-pair comparison uses exact equality after round-trip CSV parsing. All scientific values, selected parameters and search classifications match. There are 2,280 differences solely in new-encode execution counters because the fresh run did not reuse historical cached encodes. Every difference is recorded in `results/replay_differences.csv`.

## Manuscript-number audit

`results/paper_numbers.csv` records the extra manuscript quantities, sources and comparison status. The natural-image SSIM majority share is 53.5%. Paired SSIM T1-minus-C1 differences match 14.2 [13.2, 15.4] for JPEG 2000 and 2.0 [1.7, 2.3] for JPEG XL, in units of 10^-3. The minimum SSIM distance R-squared domain is aerial imagery.

One displayed number does not match: the manuscript says 2.7x for the maximum Pillow-defaults factor, whereas the maximum ratio of saved domain-quality means is 2.7649200408302144 (aerial imagery, q=30), which rounds to **2.8x**. The measurements and original manuscript were not altered to conceal this discrepancy. The individual-image maximum is recorded separately because it measures a different quantity.

The maximum q=30/q=70 reference-table R-squared change is 0.03744203710282701; all 36 domain-by-quality compression-ratio slopes are negative. No explicit corresponding claims were found in the supplied workspace manuscript, so the audit records these as absent instead of claiming a textual match.

## Evidence

Machine-readable reports are in `results/artifact_verification.json`, `results/regeneration_check.json`, `results/replay_check.json`, `results/smoke_check.json`, `results/encoder_check_verification.json`, `results/paper_numbers_check.json`, `results/privacy_check.json`, and `results/source_preservation_check.json`. Package and native codec versions are in `results/versions.txt`.

The privacy scan covers the public file set, not ignored local inputs, logs, the virtual environment, or Git author metadata. Cross-platform numerical identity has not been tested. At the time of this version 1.0 verification, no remote was created and nothing was pushed.

## Revision (version 2.0.0)

Verified on 2026-10-24 on Windows x86-64 with Python 3.11.9. A clean copy of exactly the files in this commit was used, with a new virtual environment created from `requirements.txt`. The original downloads were read-only inputs. The source workspace that produced the revision results was used only for comparison.

| Check | Result |
|---|---|
| Install pinned requirements | Passed. `pip check` reports no broken requirements, and `pip freeze` equals the 30 pins in `requirements.txt` (pytest and its dependencies merged from the revision environment) |
| Unit tests | 38 passed: the 25 existing tests and 13 revision tests, under both `python -m unittest discover -s tests` and `pytest` |
| Default `EncoderConfig` equals the published encoder | Byte-identical. The unit test compares against the literal published encoder calls. In addition, 960 encodes of the 160 real subset crops (JPEG, JPEG 2000 and JPEG XL, two settings each) match the version 1.0 `encode()` in bytes, PSNR, SSIM and size. The audit's 12 default-encoder probes reproduce the published baseline bytes |
| Rebuild revision images | All 160 subset crops and all 160 size-study images match their recorded pixel SHA-256 |
| `audit` | Passed. The subset re-selected from `data/manifest.csv` equals `data/revision/subset.csv`, and the five generated split assignments equal `data/revision/prediction_splits.csv` |
| Regenerate the supplement without images or encoding (`reporting`) | Passed in 10.2 seconds. All six table bodies (252 rows) are identical to the tables in the final supplement. All 15 regenerated result files are byte-identical to the published copies. All 44 values in `values.tex` appear verbatim in the supplement or main-text summary. All three figures have identical rendered pixels (PDF metadata can differ) |
| `validation` on the published results | Passed: 5,760 configuration, 1,920 size and 9,095 prediction rows. The baseline audits (`baseline_train_only_audit.csv`, `baseline_r2_audit.csv`, `baseline_ranking_audit.csv`) are byte-identical to the source run's, the codec settings equal `codec_details.json`, and `environment.json` is byte-identical |
| End-to-end smoke test | Passed (details below) |
| Size-study eligibility from raw downloads | Reproduced with `--reselect-size`: from all 1,009 natural and X-ray baseline crops (hashes verified), 560 eligible and 449 excluded sources and the 160-row size manifest. Identifiers, coordinates, hashes and paths equal the frozen files |
| Public-file privacy scan | No absolute paths, usernames, email addresses, credential patterns or AI attribution in the 156 committed files (CSV, JSON, TeX and PDF included) |
| Baseline preservation | No version 1.0 file under `data/`, `results/` or `paper/` changed |

### Smoke test

The smoke test ran the public command line from the original downloads: two groups per domain (eight subset images), two natural and two X-ray size groups (eight images), and an explicit five-seed split with one training and one test group per domain. Its images were freshly rebuilt into isolated directories, and every hash matched.

| Stage | Rows | Wall seconds | Process CPU seconds | Actual encodes |
|---|---|---|---|---|
| `configuration` (including image rebuild) | 288 | 150.5 | 404.6 | 3,485 |
| `size` | 96 | 65.1 | 114.1 | 858 |
| `prediction` | 1,150 (20 tasks) | 114.3 | 142.4 | 892 |
| `reporting` | 6 table bodies, 3 figures, values | 9.9 | 8.5 | 0 |

Encoding is deterministic in this environment, so the smoke rows can be checked against the published full run. All 288 configuration rows and all 96 size rows are identical to the corresponding published rows in all 35 columns apart from elapsed time and measurement origin. This covers the fresh searches for effort 3, effort 9, default transforms and 4:4:4, and the middle-slice MRI images. The prediction rows are paired across the five predictors and contain no JPEG XL X-ray q=10 row; their models were trained on the smoke split, so they are not compared with the full study. Small-sample estimates are not compared with published estimates.

The CPU-hour estimates in the README scale these process CPU times by the full run's recorded encode counts (78,325 configuration, 19,354 size and 7,180 prediction encodes).

### Differences from the source files

Numbers are unchanged. The copied files differ from the source workspace only in these ways:

- Image paths point to the public locations `data/crops/` and `data/revision_crops/`. This affects `subset.csv`, `size_manifest.csv`, `size_eligible.csv`, `mri_representative_bias.csv` and `mri_middle_slice_representativeness.csv`.
- In `protocol.json`, the search entry names `src.search.match`. Its `implementation_hashes` still record the original implementation files.
- `step_b_representativeness.json` drops a note about a local job-notification tool.
- `validation.json` is the one regenerated by this verification run. Its row counts and the hashes of the split and coefficient files equal the source run's; the hashes of the three path-updated inputs and the baseline file count (57) reflect the public layout.

Intermediate files, superseded lowest-index results, image crops, checkpoints, logs and manuscript files are not published.

### File sizes

The largest new file is `results/revision/configuration_paired.csv` (3.2 MB). Four baseline files over 10 MB (`results/pairs.csv`, `results/payload_measurements.csv`, `results/header_measurements.csv` and `results/model_evaluation.csv`) are unchanged from version 1.0.
