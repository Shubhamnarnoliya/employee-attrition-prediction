# Employee Attrition Prediction and Risk Scoring System

## Project Overview
This project uses Machine Learning to analyse employee data, predict employee attrition risk, and provide HR teams with insights to support employee retention strategies.

The system includes a Machine Learning model and an interactive Streamlit dashboard for employee risk analysis.

## Objectives
- Analyse employee attrition patterns.
- Identify important factors associated with employee attrition.
- Train and compare Machine Learning models.
- Generate employee attrition risk scores.
- Provide an interactive HR analytics dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Jupyter Notebook

## Machine Learning Algorithms
The project compares three algorithms:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

The best-performing model is selected using ROC-AUC.

## Project Structure

```text
employee-attrition-prediction/
│
├── data/
│   ├── Palo Alto Networks.csv
│   ├── processed_data.csv
│   └── model_data.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── models/
│   └── attrition_model.pkl
│
├── reports/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── employee_risk_scores.csv
│   └── test_predictions.csv
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

## Project Workflow

1. Dataset collection and loading.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis.
4. Feature engineering.
5. Train-test splitting.
6. Machine Learning model training.
7. Model evaluation and comparison.
8. Employee risk scoring.
9. Streamlit dashboard development.

## Risk Categories

| Risk Score | Category |
|---|---|
| Below 30% | Low Risk |
| 30%–60% | Medium Risk |
| Above 60% | High Risk |

The dashboard also allows the user to adjust the high-risk threshold.

## Dashboard Features

- HR overview and key performance indicators.
- Employee attrition risk distribution.
- Department-wise attrition analysis.
- Individual employee risk profiles.
- Feature importance visualisation.
- Machine Learning model performance.
- Employee risk report download in CSV format.

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd employee-attrition-prediction
```

### 2. Create a virtual environment

```bash
py -3.12 -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Jupyter Notebook

Open:

`notebooks/01_data_exploration.ipynb`

Run the notebook cells to train the models and generate the required files.

### 6. Launch the Streamlit dashboard

```bash
python -m streamlit run app/app.py
```

## Project Limitations

- Predictions are based on historical employee data.
- Risk scores are model estimates and are not guaranteed future outcomes.
- Model performance depends on dataset quality and representativeness.
- Human review is necessary before taking HR retention actions.

## Conclusion

The Employee Attrition Prediction and Risk Scoring System demonstrates how Machine Learning and data visualisation can support HR analytics. It helps identify employee attrition patterns, estimate risk scores, and provide insights that can assist organisations in developing employee retention strategies.

**Project Type:** Academic Machine Learning Project  
**Domain:** Human Resource Analytics