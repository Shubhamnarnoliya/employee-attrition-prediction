
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------
# 1. PROJECT PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "attrition_model.pkl"
DATA_PATH = BASE_DIR / "data" / "model_data.csv"
IMPORTANCE_PATH = BASE_DIR / "reports" / "feature_importance.csv"

# --------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction & Risk Dashboard")

st.write(
    "Machine Learning-Based Employee Attrition Prediction "
    "and Risk Scoring System"
)

# --------------------------------
# 3. CHECK AND LOAD FILES
# --------------------------------

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Please run the model training cells in your notebook.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.info("Please run the notebook cells that save model_data.csv.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    bundle = load_model()
    df = load_data()

    model = bundle["model"]
    features = bundle["features"]
    model_name = bundle["model_name"]

    # Check required columns
    missing_columns = [
        col for col in features if col not in df.columns
    ]

    if missing_columns:
        st.error(
            "Missing columns in dataset: "
            + ", ".join(missing_columns)
        )
        st.stop()

    # Prepare employee data
    X = df[features].copy()

    # Predict employee risk
    df["RiskScore"] = model.predict_proba(X)[:, 1] * 100

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.info(
        "Check the model file and ensure the scikit-learn "
        "version matches the version used during training."
    )
    st.stop()

# --------------------------------
# 4. SIDEBAR FILTERS
# --------------------------------

st.sidebar.header("Dashboard Filters")

threshold = st.sidebar.slider(
    "High Risk Threshold (%)",
    min_value=31,
    max_value=90,
    value=60
)

department_options = ["All"] + sorted(
    df["Department"].dropna().unique().tolist()
)

selected_department = st.sidebar.selectbox(
    "Select Department",
    department_options
)

if selected_department == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["Department"] == selected_department
    ].copy()

# Risk classification

filtered_df["RiskCategory"] = np.select(
    [
        filtered_df["RiskScore"] < 30,
        filtered_df["RiskScore"] <= threshold
    ],
    [
        "Low Risk",
        "Medium Risk"
    ],
    default="High Risk"
)

# --------------------------------
# 5. HR OVERVIEW
# --------------------------------

st.subheader("HR Overview")

total_employees = len(filtered_df)

employees_left = int(
    filtered_df["Attrition"].sum()
) if total_employees else 0

actual_attrition = (
    employees_left / total_employees * 100
    if total_employees else 0
)

high_risk_count = int(
    (filtered_df["RiskCategory"] == "High Risk").sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Employees",
    total_employees
)

col2.metric(
    "Employees Left",
    employees_left
)

col3.metric(
    "Actual Attrition Rate",
    f"{actual_attrition:.2f}%"
)

col4.metric(
    "High Risk Employees",
    high_risk_count
)

st.divider()

# --------------------------------
# 6. RISK DISTRIBUTION
# --------------------------------

st.subheader("Employee Risk Distribution")

risk_counts = (
    filtered_df["RiskCategory"]
    .value_counts()
    .reindex(
        ["Low Risk", "Medium Risk", "High Risk"],
        fill_value=0
    )
    .rename_axis("Risk Category")
    .reset_index(name="Employees")
)

fig_risk = px.pie(
    risk_counts,
    names="Risk Category",
    values="Employees",
    title="Distribution of Employee Risk Categories",
    hole=0.4
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

# --------------------------------
# 7. DEPARTMENT ANALYSIS
# --------------------------------

st.subheader("Department-wise Attrition Analysis")

department_summary = (
    filtered_df.groupby("Department")
    .agg(
        TotalEmployees=("Attrition", "count"),
        ActualAttritionRate=("Attrition", "mean"),
        AverageRiskScore=("RiskScore", "mean")
    )
    .reset_index()
)

department_summary["ActualAttritionRate"] *= 100

col1, col2 = st.columns(2)

with col1:

    fig_department = px.bar(
        department_summary,
        x="Department",
        y="ActualAttritionRate",
        title="Actual Attrition Rate by Department",
        labels={
            "ActualAttritionRate": "Attrition Rate (%)"
        }
    )

    st.plotly_chart(
        fig_department,
        use_container_width=True
    )

with col2:

    fig_average_risk = px.bar(
        department_summary,
        x="Department",
        y="AverageRiskScore",
        title="Average Predicted Risk by Department",
        labels={
            "AverageRiskScore": "Average Risk Score (%)"
        }
    )

    st.plotly_chart(
        fig_average_risk,
        use_container_width=True
    )

st.dataframe(
    department_summary.round(2),
    use_container_width=True,
    hide_index=True
)

# --------------------------------
# 8. EMPLOYEE RISK TABLE
# --------------------------------

st.subheader("Employee Risk Assessment")

display_columns = [
    "Age",
    "Department",
    "JobRole",
    "OverTime",
    "JobSatisfaction",
    "RiskScore",
    "RiskCategory"
]

risk_table = filtered_df.sort_values(
    "RiskScore",
    ascending=False
)

st.dataframe(
    risk_table[display_columns].round(2),
    use_container_width=True,
    hide_index=True
)

# --------------------------------
# 9. INDIVIDUAL EMPLOYEE PROFILE
# --------------------------------

st.subheader("Individual Employee Risk Profile")

if not filtered_df.empty:

    employee_index = st.selectbox(
        "Select Employee Record",
        filtered_df.index.tolist(),
        format_func=lambda x: f"Employee Record {x + 1}"
    )

    employee = filtered_df.loc[employee_index]

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Employee Information**")

        st.write("Age:", employee["Age"])
        st.write("Department:", employee["Department"])
        st.write("Job Role:", employee["JobRole"])
        st.write("Overtime:", employee["OverTime"])
        st.write(
            "Job Satisfaction:",
            employee["JobSatisfaction"]
        )

    with col2:

        score = float(employee["RiskScore"])

        st.metric(
            "Predicted Attrition Risk",
            f"{score:.2f}%"
        )

        if score < 30:
            st.success("Low Risk")

        elif score <= threshold:
            st.warning("Medium Risk")

        else:
            st.error("High Risk")

else:

    st.info("No employees found for this department.")

# --------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------

st.subheader("Important Attrition Factors")

if IMPORTANCE_PATH.exists():

    importance_df = pd.read_csv(IMPORTANCE_PATH)

    if {"Feature", "Importance"}.issubset(
        importance_df.columns
    ):

        importance_df = (
            importance_df.head(10)
            .sort_values(
                "Importance",
                ascending=True
            )
        )

        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Features Influencing Model Predictions"
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance file has an incorrect format."
        )

else:

    st.info("Feature importance report not found.")

# --------------------------------
# 11. MODEL PERFORMANCE
# --------------------------------

st.subheader("Machine Learning Model Performance")

st.write("Selected Model:", model_name)

model_metrics = bundle.get("metrics", {})

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

metric_cols = st.columns(5)

for col, metric_name in zip(
    metric_cols,
    metric_names
):

    value = model_metrics.get(metric_name)

    if value is not None and pd.notna(value):

        col.metric(
            metric_name,
            f"{float(value):.3f}"
        )

    else:

        col.metric(metric_name, "N/A")

# --------------------------------
# 12. DOWNLOAD REPORT
# --------------------------------

st.subheader("Download Employee Risk Report")

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Risk Assessment CSV",
    data=csv_data,
    file_name="employee_risk_report.csv",
    mime="text/csv"
)

# --------------------------------
# 13. FOOTER
# --------------------------------

st.divider()

st.caption(
    "Employee Attrition Prediction System | "
    "Academic Project | BCA"
)

st.caption(
    "Risk scores are model estimates based on historical data. "
    "They are not guaranteed predictions and should not be used "
    "as the sole basis for employment decisions. Human review "
    "and appropriate privacy safeguards are essential."
)
