# JPEG codec equivalence across imaging domains

Code and numerical results for **Complexity-Aware Parameter Equivalence Between JPEG, JPEG 2000, and JPEG XL Across Imaging Domains**, by Mahmud Hasan and Mahmoud R. El-Sakka. The study matches JPEG fidelity with JPEG 2000 compression ratios and JPEG XL distances under PSNR and SSIM for 5,970 crops from four imaging domains and nine JPEG quality levels. It measures parameter dispersion, rate savings, content-dependent prediction errors, error tails, header effects, and reference-quantizer sensitivity. Models use group-disjoint training and test sets; reported errors average domain means equally.

## What is included

- `src/`: crop reconstruction, strict codecs, search, models, statistics, header parsing, and plotting.
- `scripts/`: numbered executable stages.
- `data/manifest.csv`: source dataset/file, crop coordinates, group, channels, pixel SHA-256, and MRI orientation/window metadata.
- `data/data_exclusions.csv`: source images excluded by size, annotation availability, lung coverage, or duplicate pixels.
- `data/sanity_manifest.csv`: the separate frozen 20-image encoder-configuration sample.
- `data/replay_subset.csv`: identifiers for the 612-pair reproducibility check.
- `results/`: published search pairs, features, splits, fitted coefficients, actual model re-encoding measurements, per-file headers, robustness measurements, summaries, and version evidence.
- `paper/`: 109 value macros, three table bodies, and three PDF figures.
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

## Verification

See `VERIFICATION.md` for the fresh-environment tests, artifact comparisons, 20-image end-to-end smoke test, 612-pair replay, privacy scan, and manuscript-number mismatches. No remote is configured for this repository.
