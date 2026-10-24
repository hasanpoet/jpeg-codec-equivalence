# JPEG codec equivalence across imaging domains

Code and numerical results for **Complexity-Aware Operating-Point Equivalence of JPEG, JPEG 2000, and JPEG XL Across Imaging Domains**, by Mahmud Hasan and Mahmoud R. El-Sakka. The study matches JPEG fidelity with JPEG 2000 compression ratios and JPEG XL distances under PSNR and SSIM for 5,970 crops from four imaging domains and nine JPEG quality levels. It measures parameter dispersion, rate savings, content-dependent prediction errors, error tails, header effects, and reference-quantizer sensitivity. Models use group-disjoint training and test sets; reported errors average domain means equally.

## What is included

- `src/`: crop reconstruction, strict codecs, search, models, statistics, header parsing, and plotting.
- `scripts/`: numbered executable stages.
- `data/manifest.csv`: source dataset/file, crop coordinates, group, channels, pixel SHA-256, and MRI orientation/window metadata.
- `data/data_exclusions.csv`: source images excluded by size, annotation availability, lung coverage, or duplicate pixels.
- `data/sanity_manifest.csv`: the separate frozen 20-image encoder-configuration sample.
- `data/replay_subset.csv`: identifiers for the 612-pair reproducibility check.
- `results/`: published search pairs, features, splits, fitted coefficients, actual model re-encoding measurements, per-file headers, robustness measurements, summaries, and version evidence.
- `paper/`: 109 value macros, three table bodies, and three PDF figures.
- `data/revision/`, `results/revision/`, `paper/revision/` and `scripts/10_revision.py`: the revision's encoder-configuration, image-size and prediction-robustness experiments (see [Revision experiments](#revision-experiments)).
- `tests/`: scientific invariants and display-format tests.

No dataset images are redistributed. The MIT license applies to the code; it does not relicense third-party datasets. Source filenames are public dataset identifiers, not newly disclosed patient identifiers.

## Datasets

Download images from the original providers and follow their terms and citation requirements:

| Domain | Official source | Expected input |
|---|---|---|
| Natural photographs | [Berkeley segmentation dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) (follow BSDS500) | `BSR_bsds500.tgz` |
| Chest radiographs | [NLM Montgomery](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/index.html) and [NLM Shenzhen](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html) | `NLM-MontgomeryCXRSet.zip`, `ChinaSet_AllFiles.zip`, or extracted archive members |
| Brain MRI | [IXI T1 data](https://brain-development.org/ixi-dataset/) | `IXI-T1.tar`, or `ixi_nifti/*.nii.gz` |
| Aerial imagery | [UC Merced computer vision lab](https://vision.ucmerced.edu/datasets/) | `ucmerced/UCMerced_LandUse.zip`, or `ucmerced/Images/*/*.tif` |

IXI specifies CC BY-SA 3.0. Berkeley describes research/educational usage terms; consult its BSDS500 distribution terms. Consult NLM and UC Merced for their current usage and attribution conditions. None of these image collections is bundled here.

Place downloads under `data/downloads/`, or supply any directory with `--dataset-root`. For extracted NLM data, preserve the member paths recorded after `::` in the manifest. The fixed crop coordinates make rebuilding the published sample independent of re-running lung-mask selection; no landmarks, masks, or segmentation model are needed. Crops are 256 by 256 pixels. X-rays use integer-factor-two BOX downsampling before cropping; MRI uses the saved canonical sagittal slice and volume-window endpoints; natural/aerial images use saved native-resolution crop coordinates. Every rebuilt crop must match its recorded **pixel-byte** SHA-256 (not the compressed PNG file hash).

## Environment setup

Use Python 3.11.9. The verified environment is Windows x86-64; native codec versions are checked explicitly. Other platforms must provide wheels with the recorded native versions; cross-platform numerical identity is not assumed.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a POSIX shell, then:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements.txt` pins dependencies. `results/versions.txt` records Python, package and native codec versions. `pillow-jxl-plugin` supplies libjxl for the existing frame-header parsing API; imagecodecs supplies encoding/decoding. The header parser and encoder use separately recorded native library versions.

## Rebuild the published crops

```sh
python scripts/01_build_manifest.py --dataset-root data/downloads
```

This reconstructs and verifies the frozen manifest, rather than selecting a new sample. Images are written to ignored `data/crops/`. Source files are read only. A different crop destination can be passed with `--crops-dir` to every image-dependent stage.

## Run the full experiment

Allow roughly **25-45 CPU core-hours**, as a planning estimate rather than a benchmark from this extraction. Run the following from the repository root. These commands replace numerical outputs in this repository; keep a separate checkout if you want to retain an untouched published-results reference.

```sh
python scripts/00_encoder_check.py --dataset-root data/downloads
python scripts/01_build_manifest.py --dataset-root data/downloads
python scripts/02_search.py --workers 4
python scripts/03_psnr_models.py --workers 4
python scripts/04_ssim_models.py --workers 4
python scripts/05_headers.py --workers 4
python scripts/06_reference_robustness.py
python scripts/07_payload.py
python scripts/08_regenerate.py
```

Stages 03 and 04 encode at predicted parameters; stage 05 measures framing on retained target/equivalent files. Stage 06 recomputes q=30/q=70 reference-table DCT features and train-only fits, without new codec searches. Stage 00 reproduces its separate original three-quality, four-configuration check; its bisection settings intentionally differ from the main final-protocol search.

All stages accept `--results-dir` and `--paper-dir` for isolated outputs. Image stages also accept `--manifest` and `--crops-dir`. `--split-file` supports an explicit group-disjoint split for a small validation sample; the main experiment defaults to the published seed-0 80/20 group procedure. Fresh experiments generate their own model checkpoints; published historical checkpoints are not needed.

## Regenerate the paper without images or encoding

```sh
python scripts/08_regenerate.py
```

This uses `results/pairs.csv` **and the accompanying saved measurements**. Pairs alone cannot reconstruct DCT features, header lengths, or fidelity measured at model predictions. Regeneration recomputes summaries and 10,000-replicate group-bootstrap intervals, writes both full nine-level equivalence tables, `paper/values.tex`, all three table bodies, all three figures, and `results/paper_numbers.csv`. It performs no encoding or image access. Runtime is measured in `results/regeneration_check.json`.

`paper_numbers.csv` preserves unrounded values, displayed values, source files, domain/quality/codec identifiers, and manuscript-match status for the extra reported quantities. The current manuscript states a 2.7x maximum Pillow-defaults factor; the maximum ratio of domain-quality means in the saved sanity table is 2.7649200408302144, which rounds to 2.8x. The individual-image maximum is separately reported because it is a different estimand. The q=30/q=70 maximum R-squared change and slope signs are included, but no explicit corresponding statements were found in the supplied workspace manuscript. These discrepancies are reported rather than silently changing measurements or the manuscript.

## Important encoder settings

Pillow's JPEG 2000 defaults, `mct=0` and `irreversible=False`, must be overridden for the main experiment: use `irreversible=True` and `mct=1` for RGB (`mct=0` for grayscale). The experiment also uses one rate layer and one full-image tile. Omitting the overrides changes the color transform and wavelet, substantially increasing the number of bits needed to match JPEG PSNR in the encoder-configuration check.

JPEG uses baseline libjpeg-turbo, quality 10 through 90, 4:2:0 chroma subsampling for color, no progressive coding and no optimization. JPEG XL uses distance control, effort 7 and one encoder thread. Decoded shape and dtype must match the input. PSNR covers all channels; SSIM uses Pillow YCbCr luma for color and the unchanged grayscale channel otherwise.

## Search, statistics, and provenance

The final protocol uses geometric bisection over JPEG 2000 ratios [1.5,1000] and JPEG XL distances [0.05,25]. A detected fidelity reversal triggers 64 log-spaced evaluations over the full bracket, retaining the closest observation across bisection and scan. A scan result is accepted within 0.1 dB PSNR or 0.001 SSIM. Ordinary monotone bisection retains its closest in-range result even if its stricter tolerance is missed. Tolerance flags remain separate from acceptance. Status frequencies are reported, never stop conditions.

Only integrity failures stop processing: duplicate hashes/identifiers, group leakage, inconsistent codec round trips, crashes, corrupted or incompatible outputs. JPEG XL's X-ray q=10 cell is excluded from PSNR and SSIM model fitting and evaluation. Rate comparisons and equivalence tables still report that cell using their codec/metric acceptance rules. Rankings require both codecs to be accepted. Group-bootstrap resampling uses 10,000 replicates and seed 0; model differences are paired within domains with equal domain weights.

Published pairs use the final protocol and encoder settings, including the final nonmonotone scan fallback. `data/replay_subset.csv` identifies the 612 image-quality pairs retained from the initial search attempt for an independent fresh-search check. The replay compares numerical outcomes and search classifications; execution counters for newly performed encodes are distinguished from scientific measurements. The fresh replay reproduced all 612 pairs exactly for scientific values and search classifications (916.75 seconds); only 2,280 new-encode execution-counter cells differ because historical cached encodes were not reused. Every difference is listed in `results/replay_differences.csv`.

To repeat this check without replacing the published pairs:

```sh
python scripts/09_replay_subset.py --dataset-root data/downloads --results-dir .verification/replay --crops-dir .verification/replay_crops
```

## Revision experiments

Version 2.0.0 adds three bounded sensitivity experiments on a 160-image subset. The baseline above is frozen: none of its files change, and the new results are sensitivity estimates, not a replacement benchmark. Encoder settings, bounds, tolerances and the search are the baseline ones unless a stage says otherwise. `EncoderConfig` in `src/codecs.py` carries the varied settings, and its default is byte-identical to the baseline encoder. The `reporting` and `validation` stages perform no new encodes.

### Stages

| Stage | What it does | Estimated CPU core-hours |
|---|---|---|
| `audit` | Hashes the baseline files, re-selects the subset from `data/manifest.csv` and requires it to equal `data/revision/subset.csv`, re-encodes one image per domain with the default encoder against the published bytes, checks held-out evaluation pairing, and checks the protocol against `data/revision/protocol.json`. | < 0.1 |
| `configuration` | For each subset image at JPEG q=10/50/90, matches PSNR and SSIM under: the selected settings (the baseline search is reproduced byte for byte from `results/pairs.csv`), JPEG XL effort 3 and 9, JPEG 2000 with Pillow's default transforms (reversible wavelet, no RGB transform), and a 4:4:4 JPEG reference (color images only). Records total-file, header and payload bytes. | about 2.5 |
| `size` | Repeats the selected-settings matching on 40 natural and 40 X-ray pairs: a 256 crop and a larger field from the same source that contains it pixel for pixel. | about 0.7 |
| `prediction` | Five group-disjoint 80/20 splits (seeds 0-4; 32 training and 8 test groups per domain). Fits T0/T1/C0/C1 on training groups only, with 1,000 training-group bootstrap intervals for the coefficients, then encodes every eligible held-out image at all four predictions. The JPEG XL X-ray q=10 cell stays excluded. | about 0.3 |
| `reporting` | Summaries, paired tables, 10,000-replicate paired group-bootstrap differences, coefficient stability, the asymmetry gate, the six supplement table bodies, three figures and `paper/revision/values.tex`. | < 0.01 |
| `validation` | Coverage, pairing, leakage, provenance and pixel-hash checks. Requires the q=50 asymmetry gate to pass and every size pair to be pixel-nested. Recomputes baseline coefficients, held-out R-squared and codec-ranking counts from the raw baseline tables, and requires the runtime codec settings to equal `data/revision/codec_details.json`. Writes `validation.json` and `environment.json`. | < 0.01 |

The whole revision takes about 3.5 CPU core-hours, roughly 1 to 1.5 hours of wall time with `--workers 4`. These figures are extrapolated from the smoke test in `VERIFICATION.md`: its measured process CPU time per image or encode, scaled by the encode counts recorded in the full run. They are planning estimates, not benchmarks.

### Commands

Rebuild the 320 revision images from the original downloads, each verified against its recorded pixel SHA-256, then run every stage:

```sh
python scripts/10_revision.py all --dataset-root data/downloads --workers 4
```

The same script runs one stage at a time: `audit`, `configuration`, `size`, `prediction`, `reporting` or `validation`. `--dataset-root` is needed only once; images go to the ignored `data/crops/` and `data/revision_crops/`. Image stages checkpoint every image under `results/revision/checkpoints/`, keyed by protocol, source pixels, configuration and split, and resume after interruption. Running a stage replaces its published outputs in `results/revision/`; use `--results-dir` and `--paper-dir` for an isolated run. A smaller run can use `--subset`, `--size-manifest` and an explicit `--split-file` (columns `seed,domain,group,split`), as in the smoke test.

To regenerate the supplement tables and figures from the published measurements, without images or encoding:

```sh
python scripts/10_revision.py reporting
```

X-ray size eligibility depends on lung masks: the manual masks inside the Montgomery archive, and Shenzhen lung landmarks, which this repository does not redistribute. The frozen coordinates in `data/revision/size_manifest.csv` remove that dependency. With the landmarks available, `--reselect-size LANDMARKS_DIR` re-derives eligibility, exclusions and selection into the results directory and requires them to equal the frozen manifests.

### Subset and the MRI middle-slice rule

Groups in each domain are ordered by median baseline DCT entropy (ties by group), split into rank quartiles, and ten per quartile are drawn with NumPy seed 0. This gives 40 independent groups per domain, with one image representing each group. For MRI, that image is the lower-middle retained slice by canonical slice index (`middle_slice_by_canonical_index_lower_middle`; the image identifier breaks ties). The other domains use the lexicographically first image identifier.

The first revision run used the lexicographically first image for MRI too, which is the lowest slice index. For every one of the 40 subjects, that picked slice 20, the outermost of the subject's five retained slices. These are low-content edge slices, with a median DCT entropy ratio of 0.536 to the subject median (`results/revision/mri_representative_bias.csv`). Such a slice does not represent its subject, so the rule was replaced. The middle-slice representatives (slices 72-74, none at an edge) are mildly biased high, with a median ratio of 1.133 (IQR 1.111-1.145; `mri_middle_slice_representativeness.csv`). This was accepted because their q=50 PSNR-matched median settings are within 21% (JPEG 2000) and 3% (JPEG XL) of the full-dataset MRI medians (`step_b_representativeness.json`). The selected groups did not change, and the non-MRI images and the size study are unaffected. The superseded lowest-index results are not published; the bias audit records why they were replaced.

In the published `configuration_results.csv` and `prediction_results.csv`, `measurement_origin` records how each row was obtained. MRI rows were measured afresh for the corrected subset. Non-MRI rows were carried over from the first run only where the source pixel hash was identical, and for prediction also the predicted setting; `prediction_measurement_origins.csv` counts them. A fresh run measures every row. Encoding is deterministic in the pinned environment, so the values are the same.

For X-ray size pairs, no original 256 crop lies inside a fully lung-covered 512 square (509 of 509 fail; `size_exact_original_exclusions.csv`). The pairs are therefore new: the fully covered 512 square closest to the original crop centre, and its central 256 square (`crop256_new`), from 60 eligible patients (`size_eligible.csv`, `size_exclusions.csv`, `size_fallback.json`). Natural pairs are the original crop and the full BSDS image. Larger fields add content, so this measures size and content together, not an isolated scale effect.

### Outputs and the supplement

`data/revision/` holds the frozen inputs as identifiers, coordinates and pixel hashes only:
- `subset.csv`: the 160 images, with the middle-slice MRI rule.
- `size_manifest.csv`, with `size_eligible.csv`, `size_exclusions.csv`, `size_exact_original_exclusions.csv` and `size_fallback.json`.
- `prediction_splits.csv`: the split assignments for seeds 0-4.
- `protocol.json` and `codec_details.json`. `protocol.json` keeps the hashes of the original implementation files that produced the published results.

| Supplement item | `paper/revision/` file | Computed from (`results/revision/`) |
|---|---|---|
| Figure S1, configuration sensitivity | `figS1_configuration.pdf` | `configuration_summary.csv` |
| Table S1, configuration summary at q=50 | `tableS1_body.tex` | `configuration_summary.csv` |
| Table S2, between-domain max/min by quality | `tableS2_body.tex` | `configuration_between_domain_by_q.csv` (full-dataset column from the baseline `results/pairs.csv`) |
| Figure S2, size sensitivity | `figS2_size.pdf` | `size_summary.csv` |
| Table S3, size summary at q=50 | `tableS3_body.tex` | `size_summary.csv` |
| Figure S3, prediction robustness | `figS3_prediction.pdf` | `prediction_equal_domain.csv` |
| Table S4, held-out errors by split | `tableS4_body.tex` | `prediction_equal_domain.csv` |
| Table S5, paired error differences | `tableS5_body.tex` | `prediction_differences.csv` |
| Table S6, C1 coefficient stability | `tableS6_body.tex` | `C1_coefficient_summary.csv` |
| In-text values | `values.tex` | `configuration_between_domain_by_q.csv`, `asymmetry_gate.json`, `configuration_asymmetry_by_q.csv`, `mri_representative_bias.csv`, `mri_middle_slice_representativeness.csv`, `step_b_representativeness.json`, `size_summary.csv`, `prediction_differences.csv`, `prediction_equal_domain.csv`, `worked_example.json` |

Upstream measurements:
- `configuration_results.csv` feeds `configuration_summary.csv`, `configuration_paired.csv` (each alternative against the selected setting) and `configuration_asymmetry_by_q.csv`.
- `size_results.csv` feeds `size_summary.csv`, `size_paired.csv` and `size_paired_summary.csv`.
- `prediction_results.csv` feeds `prediction_summary.csv` (per domain), `prediction_equal_domain.csv` and `prediction_differences.csv`.
- `prediction_coefficients.csv` feeds `coefficient_stability.csv` (all predictors) and `C1_coefficient_summary.csv`.

`validation.json` and `environment.json` record the validation run and its Python environment.

## Verification

See `VERIFICATION.md` for the fresh-environment tests, artifact comparisons, 20-image end-to-end smoke test, 612-pair replay, privacy scan, manuscript-number mismatches, and the revision checks. The public repository is https://github.com/hasanpoet/jpeg-codec-equivalence; version 2.0.0 is tagged `v2.0.0`.
