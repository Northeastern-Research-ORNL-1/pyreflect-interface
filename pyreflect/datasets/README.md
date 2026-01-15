## Dataset Overview

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
    normalization_stats: datasets/normalization_stat.npy
```

Notes:
- Keep `normalization_stat.npy` paired with the model to preserve the original scaling.
- NR curves are log10-transformed on the y-axis during preprocessing; use the same stats for any new data to avoid drift.
