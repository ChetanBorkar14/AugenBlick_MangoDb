import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from components.reports import extract_data_from_notebook  # Ensure this function exists

# Define the correct path to main.ipynb
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "main.ipynb")

def display_summary_table(data, column_name):
    """Displays summary statistics for a given numerical column."""
    if data is not None and not data.empty:
        summary = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'IQR', 'Min', 'Max', 'Count'],
            'Value': [
                data.mean(), data.median(), data.std(), 
                data.quantile(0.75) - data.quantile(0.25),  # IQR
                data.min(), data.max(), data.count()
            ]
        })

        st.subheader(f'Summary Statistics: {column_name}')
        st.table(summary)

        # 📊 Add histogram plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data, kde=True, bins=20, ax=ax)
        ax.set_title(f"Distribution of {column_name}")
        st.pyplot(fig)

    else:
        st.warning(f"No data available for {column_name}.")

def display_covariate_balance(balance_data):
    """Displays covariate balance check results."""
    if balance_data:
        balance_table = pd.DataFrame(balance_data)

        st.subheader('Covariate Balance Check')
        st.dataframe(balance_table.style.applymap(
            lambda x: 'background-color: #ffadad' if isinstance(x, (int, float)) and abs(x) > 0.1 else '',
            subset=['Unweighted_SMD', 'Weighted_SMD']
        ))

        # 📊 Add bar chart for covariate balance
        fig, ax = plt.subplots(figsize=(8, 4))
        balance_table.plot(kind="bar", x="Confounder", y=["Unweighted_SMD", "Weighted_SMD"], ax=ax)
        ax.set_title("Covariate Balance (Before & After Weighting)")
        st.pyplot(fig)

    else:
        st.warning("No covariate balance data available.")

def display_bootstrap_intervals(intervals):
    """Displays bootstrap confidence intervals for treatment effects."""
    if intervals:
        intervals_table = pd.DataFrame(intervals, columns=['Parameter', 'Lower Bound', 'Upper Bound'])

        st.subheader('Bootstrap Confidence Intervals')
        st.dataframe(intervals_table)

        # 📊 Plot confidence intervals
        fig, ax = plt.subplots(figsize=(6, 4))
        for _, row in intervals_table.iterrows():
            ax.plot([row['Lower Bound'], row['Upper Bound']], [row['Parameter'], row['Parameter']], marker='o')
        ax.set_title("Bootstrap Confidence Intervals")
        st.pyplot(fig)

    else:
        st.warning("No bootstrap intervals available.")

def display_missing_data(df):
    """Displays missing value percentages for each column."""
    missing_data = df.isnull().sum() / len(df) * 100
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if not missing_data.empty:
        st.subheader("Missing Data Overview")
        st.dataframe(missing_data.to_frame("Missing %"))
        
        # 📊 Add bar plot for missing data
        fig, ax = plt.subplots(figsize=(6, 3))
        missing_data.plot(kind="bar", ax=ax, color="red")
        ax.set_title("Missing Data Percentage")
        st.pyplot(fig)
    else:
        st.success("No missing data detected!")

def download_report(df):
    """Allows users to download the dataset as a CSV file."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Data as CSV", csv, "report.csv", "text/csv")

def display_tables(ns):
    """Wrapper function to display all tables dynamically from notebook data."""
    st.write("### Data Overview")
    
    df = ns.get("df")
    if df is not None and not df.empty:
        st.write(df.head())

        # 🔍 Allow users to select a column for summary stats
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_columns:
            selected_col = st.selectbox("Select a column to view summary statistics:", numeric_columns)
            display_summary_table(df[selected_col], selected_col)

        # 🔎 Check for missing data
        display_missing_data(df)

    else:
        st.warning("Dataset is empty.")

    # 🧪 Covariate balance
    balance_data = ns.get("balance_data")
    if balance_data:
        display_covariate_balance(balance_data)
    
    # 📈 Bootstrap confidence intervals
    intervals = ns.get("bootstrap_intervals")
    if intervals:
        display_bootstrap_intervals(intervals)

    # 📂 Download Report
    if df is not None and not df.empty:
        download_report(df)

# Example usage
if __name__ == "__main__":
    try:
        ns = extract_data_from_notebook(NOTEBOOK_PATH)  # Fetch data dynamically
        display_tables(ns)
    except Exception as e:
        st.error(str(e))
