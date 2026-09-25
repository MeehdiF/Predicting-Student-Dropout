# Student Dropout Prediction

Explore classification models for identifying students who may be at higher risk of dropping out.

## Why this project

This repository is part of my practical machine-learning portfolio. It focuses on a complete, understandable workflow rather than claiming production readiness.

## Dataset

The repository includes `student-por.csv`. Document the dataset source and target definition before using this project in a formal setting.

## Approach

Categorical preprocessing, scaling where appropriate, Logistic Regression, Decision Trees, and Support Vector Machines with model evaluation.

### Features

Academic, demographic, and personal characteristics available in the student-performance data.

## Evaluation and current result

Accuracy, F1 score, Jaccard score, classification reports, and confusion matrices are used in the notebooks. Report class balance and minority-class recall before making any risk-related interpretation.

## Run locally

```bash
git clone https://github.com/MeehdiF/Predicting-Student-Dropout.git
cd Predicting-Student-Dropout
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python they_will_fail_logistic.py
python they_will_fail_Decision_trees.py
python they_will_fail_SVM.py
```

For notebook exploration, open the `.ipynb` file with Jupyter after installing the same dependencies.

## Limitations and next steps

Educational risk prediction is sensitive and should not be used for high-impact decisions without careful validation, fairness analysis, and stakeholder review. The current project should document leakage checks, split strategy, and whether the target is defined before or after the student outcome.

## Repository structure

- `README.md` — project context and reproducibility notes
- `requirements.txt` — Python dependencies used by the scripts
- `.ipynb` / `.py` files — analysis and model experiments

## License

See [`LICENSE`](LICENSE). Check the dataset's own terms separately; repository code licensing does not automatically license bundled data.
