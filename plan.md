# PLAN.md

  ## Objective
  Wire uploaded `.npy` and `.pth` files into `settings.yml` and
  runtime so real‑data runs use the uploaded assets correctly.

  ## Context
  - UI already uploads `.npy`, `.pth`, `.pt`, and
  `settings*.yml/.yaml`.
  - Backend stores uploads under `src/backend/data/` (models +
  curves).
  - Real‑data mode is not wired yet; uploads are stored but not used.

  ## Tasks
  1. Upload intake + validation
     - Classify uploads into: NR train, SLD train, experimental NR,
  normalization stats, chi inputs, model weights by having user select the type of file they are uploading, or use the one provided by PyReflect.
     - Validate file types and shapes (NR/SLD: `(N, 2, L)`; stats:
  dict; model: `.pth/.pt`).
     - Persist files to `data/curves`, `data/curves/expt`, `data/
  models` and return metadata.

  2. Settings writer/merger
     - Update `src/backend/settings.yml` with the uploaded file
  paths.
     - Ensure keys match pyreflect expectations for NR→SLD and
  SLD→Chi flows.
     - Preserve user‑edited values not related to uploads.

  3. Real‑data execution path
     - In `/api/generate/stream`, add a “real‑data” branch that reads
  `settings.yml`.
     - Load `.npy` via `NRSLDDataProcessor` (or `SLDChiDataProcessor`
  for chi prediction).
     - Train/load model and run inference using the uploaded files.

  ## Success Criteria
  - Uploading valid files updates `settings.yml` and `/api/status`
  reflects those paths.
  - Real‑data runs consume uploaded `.npy/.pth` without synthetic
  generation.
  - Invalid file types/shapes fail fast with clear, actionable
  errors.

  ## Additional Notes: Dataset Overview from PyReflect

    The data files is too large to be put in repo.

    ### Contents
    - `curves/`: synthetic 5-layer NR/SLD pairs used in examples.
    - `nr_5_layers.npy` / `sld_5_layers.npy`: full generated set.
    - `X_train_5_layers.npy` / `y_train_5_layers.npy`: training split.
    - `X_test_5_layers.npy` / `y_test_5_layers.npy`: hold-out split for evaluation.
    - `combined_nr.npy` / `combined_sld.npy`: merged NR/SLD pairs generated via `NRSLDDataGenerator`; can be used as an alternative training set.
    - `combined_expt_denoised_nr.npy`: 8 denoised experimental NR curves ready for inference.
    - `normalization_stat.npy`: min/max stats (NR + SLD) computed from the training data; reuse these for normalization/denormalization during inference.
    - `trained_nr_sld_model_no_dropout.pth`: pretrained CNN checkpoint for NR->SLD (12 layers, dropout=0.0). If your config expects `trained_nr_sld_model.pth`, point it to this file or rename it.

    ### Hooking these files into `examples/settings.yml`
    Set paths relative to the project root:
    ```yaml
    nr_predict_sld:
    file:
        nr_train: datasets/curves/X_train_5_layers.npy
        sld_train: datasets/curves/y_train_5_layers.npy
        experimental_nr_file: datasets/combined_expt_denoised_nr.npy
    models:
        model: datasets/trained_nr_sld_model_no_dropout.pth
        
    Notes:
    - Keep `normalization_stat.npy` paired with the model to preserve the original scaling.
    - NR curves are log10-transformed on the y-axis during preprocessing; use the same stats for any new data to avoid drift.

- NR→SLD: pick nr_train.npy, sld_train.npy, normalization_stat.npy,
    and a .pth model file.
  - SLD→Chi: pick model_sld_file.npy, model_chi_params_file.npy, and
    model_experimental_sld_profile.npy.

     - python -m pyreflect run ... reads settings.yml
  - It resolves the file paths from the YAML
  - Then it loads .npy and .pth only when the specific workflow is
    invoked

support both by adding a mode toggle in your backend
  request (and UI), then route to the correct branch. Think of it
  as two switches:

  1. Workflow: nr_sld vs sld_chi
  2. Mode (NR→SLD only): train vs infer

  NR → SLD (toggle: train vs infer)

  - train requires:
      - nr_predict_sld.file.nr_train → .npy (N,2,L) Q/R
      - nr_predict_sld.file.sld_train → .npy (N,2,L) z/rho
      - nr_predict_sld.models.model → output .pth path
      - nr_predict_sld.models.normalization_stats → output .npy
        path
  - infer requires:
      - nr_predict_sld.file.experimental_nr_file → .npy (N,2,L) Q/
        R
      - nr_predict_sld.models.model → existing .pth
      - nr_predict_sld.models.normalization_stats → existing
        stats .npy

  Example settings.yml keys to wire

  nr_predict_sld:
    file:
      nr_train: data/curves/nr_train.npy
      sld_train: data/curves/sld_train.npy
      experimental_nr_file: data/curves/expt/my_expt_nr.npy
    models:
      model: data/models/my_model.pth
      normalization_stats: data/normalization_stat.npy

  Backend routing (simplified)

  - If mode == "train" → load nr_train/sld_train → train → save
    model + stats
  - If mode == "infer" → load experimental_nr_file + model + stats
    → predict

  If you want, tell me where you want the toggle (API body,
  settings.yml flag, or UI-only), and I’ll outline the exact
  wiring.