# Credit Card Fraud Detection
## Introduction
This project demonstrates the full lifecycle of a machine learning project.

To keep the project centralized, I have decided to organize put all components of the project (model development, deployment, and monitoring) into a single repository.

I aim to follow best practices, and use de-facto standard or popular open source tools. 

I cover:
- Problem definition
- Exploratory data analysis
- Initial baseline & sanity checks
- Model development with experiment tracking in MLFlow
- Deployment: batch prediction
- Monitoring with Evidently
- Orchestration with Prefect

I plan to add:
- infrastructure setup (terraform + localstack)
- CI/CD
- alerts & automatic re-training

## Problem Definition
The goal is to detect fraud in credit card transactions. As the running example, I use real anonymized transactions [data](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The data is labelled, so a supervised classification model is the appropriate method.

The proportion of frauds (positive class) is very small (0.17%), i.e., the data is very imbalanced, so using accuracy as the metric is out of the question. 

Furthermore, we want to minimize both false negatives (undetected frauds) and false positives (non-fraud transactions flagged as fraud), since both affect customer experience and cost resources to handle.

Therefore, an appropriate metric is the F1-score, which balances both precision and recall.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         fraud_detector and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── fraud_detector   <- Source code for use in this project.
```

--------

