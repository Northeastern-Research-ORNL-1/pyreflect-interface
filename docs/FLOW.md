# Pyreflect Pipeline Flow

This document describes the data flow and phases of the synthetic training pipeline.

## Overview

```mermaid
flowchart TD
    subgraph Input["📥 Input"]
        UI["Web Interface"]
        Layers["Film Layers + Bounds"]
        Params["Generator & Training Params"]
    end

    subgraph Generation["🔬 Phase 1: Data Generation"]
        RDG["ReflectivityDataGenerator"]
        NR["NR Curves<br/>(N × 2 × 308)"]
        SLD["SLD Profiles<br/>(N × 2 × 900)"]
    end

    subgraph Preprocessing["⚙️ Phase 2: Preprocessing"]
        Log["Log Transform (NR)"]
        Norm["Normalization"]
        Split["Train/Val/Test Split"]
    end

    subgraph Training["🧠 Phase 3: Training"]
        CNN["CNN Model"]
        Loss["MSE Loss"]
        Opt["Adam Optimizer"]
    end

    subgraph Saving["💾 Phase 4: Model Saving"]
        Local["Local .pth File"]
        HF["Hugging Face Hub"]
        Mongo["MongoDB History"]
    end

    subgraph Inference["🎯 Phase 5: Inference"]
        TestNR["Test NR Curve"]
        PredSLD["Predicted SLD"]
        CompNR["Computed NR"]
    end

    subgraph Output["📊 Output"]
        Result["GenerateResponse"]
        Charts["NR/SLD/Training Charts"]
        Metrics["MSE, R², MAE"]
    end

    UI --> Layers --> RDG
    UI --> Params --> RDG
    RDG --> NR --> NPY_NR["💡 Optional: Save nr_train.npy"]
    RDG --> SLD --> NPY_SLD["💡 Optional: Save sld_train.npy"]
    NR --> Log --> Norm --> Split --> CNN
    SLD --> Norm
    CNN <--> Loss
    Loss <--> Opt
    CNN --> Local
    Local --> HF
    CNN --> TestNR --> PredSLD --> CompNR
    PredSLD --> Metrics
    CompNR --> Result
    Metrics --> Result
    Result --> Mongo
    Result --> Charts
```

---

## Phase Details

### Phase 1: Data Generation

The `ReflectivityDataGenerator` from pyreflect creates synthetic NR curves and corresponding SLD profiles.

```mermaid
flowchart LR
    subgraph Inputs
        LD["layer_desc<br/>(8 layers)"]
        LB["layer_bound<br/>(19 constraints)"]
        NC["num_curves<br/>(1000-100000)"]
    end

    subgraph Generator
        RDG["ReflectivityDataGenerator"]
        Refl1d["refl1d physics engine"]
    end

    subgraph Outputs
        NR["NR Array<br/>Shape: (N, 2, 308)<br/>• Q values<br/>• Reflectivity"]
        SLD["SLD Array<br/>Shape: (N, 2, 900)<br/>• Z depth<br/>• SLD values"]
    end

    LD --> RDG
    LB --> RDG
    NC --> RDG
    RDG --> Refl1d --> NR
    Refl1d --> SLD

    NR -.->|"💾 Optional"| NPY1[("nr_train.npy")]
    SLD -.->|"💾 Optional"| NPY2[("sld_train.npy")]
```

> **💡 Note:** The generated `.npy` files can optionally be saved for reuse in "Real Data" mode or external training.

---

### Phase 2: Preprocessing

Raw curves are normalized for training.

| Step             | NR Curves                    | SLD Profiles       |
| ---------------- | ---------------------------- | ------------------ |
| 1. Transform     | `log10(clip(R, 1e-8))`       | None               |
| 2. Compute Stats | `min/max` per axis           | `min/max` per axis |
| 3. Normalize     | Min-Max to [0,1]             | Min-Max to [0,1]   |
| 4. Split         | 80% train, 10% val, 10% test | Same indices       |

---

### Phase 3: Training

```mermaid
flowchart TB
    subgraph Model["CNN Architecture"]
        Input["NR (1, 308)"]
        Conv["Conv1d Layers<br/>(configurable depth)"]
        Drop["Dropout"]
        FC["Fully Connected"]
        Output["SLD (2, 900)"]
    end

    subgraph Loop["Training Loop"]
        Epoch["For each epoch"]
        Forward["Forward Pass"]
        MSE["MSE Loss"]
        Backward["Backward Pass"]
        Update["Optimizer Step"]
        Val["Validation Loss"]
    end

    Input --> Conv --> Drop --> FC --> Output
    Epoch --> Forward --> MSE --> Backward --> Update --> Val
    Val -->|"Next Epoch"| Epoch

    Val -->|"Every N epochs"| Checkpoint["☁️ HF Checkpoint"]
```

**Training Parameters:**

- `epochs`: Number of training epochs (default: 10)
- `batchSize`: Samples per batch (default: 32)
- `layers`: CNN depth (default: 12)
- `dropout`: Regularization (default: 0.0)
- `learningRate`: 0.001 (Adam)
- `weightDecay`: 0.0001

---

### Phase 4: Model Saving

```mermaid
flowchart TD
    Model["Trained Model"]
    CPU["Move to CPU"]

    subgraph Storage["Storage Options"]
        direction LR
        Local["Local .pth<br/>(default)"]
        HF["Hugging Face Hub<br/>(if configured)"]
    end

    subgraph History["History Tracking"]
        Mongo["MongoDB<br/>(generations collection)"]
    end

    Model --> CPU --> Local
    Local -->|"Upload + Verify"| HF
    HF -->|"Delete local"| Cleanup["🗑️ Cleanup"]
    Model --> Mongo
```

**Hugging Face Bundle Structure:**

Each training run creates a folder on HF with all artifacts:

```
models/{model_id}/
├── {model_id}.pth     # Trained CNN model weights
├── nr_train.npy       # NR curves (N × 2 × 308)
└── sld_train.npy      # SLD profiles (N × 2 × 900)
```

The `.npy` files are uploaded immediately after data generation (before training begins), ensuring the training data is preserved even if training fails.

---

### Phase 5: Inference

```mermaid
flowchart LR
    TestNR["Test NR Curve<br/>(from split)"]
    Model["Trained CNN"]
    PredSLD["Predicted SLD"]
    Denorm["Denormalize"]
    CompNR["compute_nr_from_sld()"]

    TestNR --> Model --> PredSLD --> Denorm
    Denorm --> CompNR
    Denorm --> Metrics["MSE, R², MAE"]

    subgraph Result["Result Object"]
        NRData["nr: {q, groundTruth, computed}"]
        SLDData["sld: {z, groundTruth, predicted}"]
        TrainData["training: {epochs, losses}"]
        Chi["chi: sampled comparison"]
        Met["metrics: {mse, r2, mae}"]
    end

    CompNR --> NRData
    Denorm --> SLDData
    Metrics --> Met
```

---

## Data Shapes Reference

| Array            | Shape             | Description                |
| ---------------- | ----------------- | -------------------------- |
| `nr_curves`      | `(N, 2, 308)`     | Q values + Reflectivity    |
| `sld_curves`     | `(N, 2, 900)`     | Z depth + SLD values       |
| `normalized_nr`  | `(N, 1, 308)`     | Y-channel only, normalized |
| `normalized_sld` | `(N, 2, 900)`     | Both channels, normalized  |
| `model output`   | `(batch, 2, 900)` | Predicted SLD profile      |

---

## Optional: Save .npy Files

The synthetic generation creates NR/SLD arrays in memory. To save them for later reuse:

### Current Behavior

- Arrays are generated → used for training → **discarded**
- Only the trained model is persisted

### Proposed Enhancement

Add a `saveTrainingData` option to the generate request:

```json
{
  "layers": [...],
  "generator": { "numCurves": 1000, ... },
  "training": { "epochs": 10, ... },
  "saveTrainingData": true  // NEW: Save .npy files
}
```

This would:

1. Save `nr_train.npy` and `sld_train.npy` to a user-specific directory
2. Upload to Hugging Face Hub (if configured)
3. Return download URLs in the response

**Use cases:**

- Reuse training data across multiple experiments
- Share datasets between team members
- Use in "Real Data" mode with pre-generated synthetic data
