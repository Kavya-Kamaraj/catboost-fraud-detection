# Identity Risk CatBoost Model

This project trains a CatBoost binary classification model using identity risk and identity match features. The package includes a notebook version of the training workflow, command-line scripts for training/evaluation/inference, saved model artifacts, and a simple flowchart.

## Project Structure

```text
identity_risk_catboost_public/
├── models/
│   └── catboost_identity_risk_model.cbm
├── outputs/
│   ├── model_evaluation/
│   └── inference_predictions.csv
├── CATBOOST_Identity_Risk_Training.ipynb
├── FLOWCHART.md
├── generate_model_outputs.py
├── inference_identity_risk.py
└── train_identity_risk_catboost.py
```

## Input Schema

The target column is:

```text
target_label
```

Accepted target values include `Y`, `N`, `1`, `0`, `true`, `false`, `yes`, and `no`. They are normalized to `Y` and `N` during training and evaluation.

Training, evaluation, and inference inputs must use the same feature schema as the trained CatBoost model.

## Install Requirements

Use an environment with these Python packages installed:

```bash
pip install -r requirements.txt
```

## Train The Model

Run:

```bash
python train_identity_risk_catboost.py
```

The script will:

```text
load input data -> clean labels/text -> split train/test -> train CatBoost -> print AUC/accuracy -> save model
```

The saved model path is:

```text
models/catboost_identity_risk_model.cbm
```

The training script uses GPU CatBoost by default:

```python
task_type="GPU"
```

If GPU is not available, change it to:

```python
task_type="CPU"
```

## Generate Evaluation Outputs

After training, run:

```bash
python generate_model_outputs.py
```

This creates:

```text
outputs/model_evaluation/threshold_analysis.csv
outputs/model_evaluation/threshold_graph.png
outputs/model_evaluation/confusion_matrix.csv
outputs/model_evaluation/confusion_matrix.png
outputs/model_evaluation/metrics.csv
```

The threshold graph shows the trade-off between false negatives, false positives, true negatives, true positives, and the FP/TP ratio.

## Run Inference

Run inference on a CSV that matches the trained model feature schema:

```bash
python inference_identity_risk.py \
  --model-path models/catboost_identity_risk_model.cbm \
  --input-path path/to/input.csv \
  --output-path outputs/inference_predictions.csv \
  --threshold 0.55
```

The output file includes:

```text
model_score
model_prediction
```

## Notebook

The notebook version is:

```text
CATBOOST_Identity_Risk_Training.ipynb
```

It follows the same training flow as the Python script.

## Flowchart

The model flow is documented in:

```text
FLOWCHART.md
```

High-level flow:

```text
Input CSV -> Clean Data -> Feature Split -> Train/Test Split -> CatBoost Training -> Model Save -> Evaluation/Inference
```

## Notes

- Only identity risk and identity match columns are used as model features.
- Evaluation and inference scripts read the trained model feature schema directly from the `.cbm` file.
