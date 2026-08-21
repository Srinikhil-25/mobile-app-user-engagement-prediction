# Mobile App User Engagement Prediction

A Business Analytics mini-project that uses mobile application event logs to analyze user engagement and predict whether a user will be active during the following 7 days.

## Project Objective

The project answers two business questions:

1. Can historical app activity be used to predict 7-day future user activity?
2. Which behavioral factors are associated with continued engagement?

## Workflow

Raw Event Logs
→ Data Cleaning
→ User-Level Feature Engineering
→ 7-Day Target Creation
→ Exploratory Analysis
→ Statistical Analysis
→ Predictive Modeling
→ Model Evaluation
→ Dashboard

## Target

`active_next_7_days`

- `1`: user is active during the following 7 days
- `0`: user is not active during the following 7 days

## Main Features

- Total events
- Unique sessions
- Active days
- Recency in days
- Events per session
- Device operating system
- Location country

## Models

- Logistic Regression
- Random Forest

## Repository Structure

```text
mobile-app-user-engagement-prediction/
├── data/
│   ├── mobile_app_interactions.csv
│   └── README.md
├── src/
│   └── pipeline.py
├── dashboard/
│   └── app.py
├── results/
├── figures/
├── run_project.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the complete pipeline

```bash
python run_project.py
```

This creates the model-ready dataset, model comparison results, confusion matrix, ROC curve, and precision-recall curve.

### 5. Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Important Methodology Note

The prediction target is created using a fixed historical cutoff and a complete 7-day future window. Records from invalid/future-anomalous dates are excluded from the analytical period so that the temporal logic is not distorted.

## Academic Use

This repository supports the Business Analytics Laboratory mini-project report. The dataset is synthetic and intended for academic analysis.
