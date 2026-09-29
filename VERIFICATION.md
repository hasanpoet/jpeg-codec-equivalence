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

The privacy scan covers the public file set, not ignored local inputs, logs, the virtual environment, or Git author metadata. Cross-platform numerical identity has not been tested. No remote was created and nothing was pushed.
