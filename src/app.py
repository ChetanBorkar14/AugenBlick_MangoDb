import streamlit as st
from components.charts import display_charts
from components.reports import generate_reports, extract_data_from_notebook
from components.tables import display_tables
from components.what_if import display_what_if_interface
from models.model_validation import validate_model
import pandas as pd
import os

# Fix file path handling
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Go one level up
DATA_PATH = os.path.join(BASE_DIR, "diabetes.csv")
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "main.ipynb")

def main():
    st.title("Data Analysis and Simulation App")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a section:", ["Home", "Charts", "Reports", "Tables", "What-If Simulation", "Model Validation"])
    
    if options == "Home":
        st.write("Welcome to the Data Analysis and Simulation App!")
        st.write("Use the sidebar to navigate through different sections.")
    
    elif options == "Charts":
        st.subheader("Data Visualizations")
        st.write(f"Loading data from: {DATA_PATH}")  # Debugging information
        try:
            data = pd.read_csv(DATA_PATH)
            display_charts(data)
        except FileNotFoundError:
            st.error(f"File not found: {DATA_PATH}")
    
    elif options == "Reports":
        st.subheader("Analysis Reports")
        st.write(f"Loading notebook from: {NOTEBOOK_PATH}")  # Debugging information
        try:
            ns = extract_data_from_notebook(NOTEBOOK_PATH)
            generate_reports(ns)
        except FileNotFoundError:
            st.error(f"File not found: {NOTEBOOK_PATH}")
    
    elif options == "Tables":
        st.subheader("Summary Tables")
        st.write(f"Loading notebook from: {NOTEBOOK_PATH}")  # Debugging information
        try:
            ns = extract_data_from_notebook(NOTEBOOK_PATH)  # Fetch data dynamically
            display_tables(ns)
        except Exception as e:
            st.error(str(e))
    
    elif options == "What-If Simulation":
        st.subheader("What-If Scenario Simulation")
        st.write(f"Loading data from: {DATA_PATH}")  # Debugging information
        try:
            data = pd.read_csv(DATA_PATH)
            display_what_if_interface(data)
        except FileNotFoundError:
            st.error(f"File not found: {DATA_PATH}")
    
    elif options == "Model Validation":
        st.subheader("Model Validation")
        predictions = [0, 1, 0, 1, 0, 1, 0, 1]
        ground_truth = [0, 1, 0, 1, 0, 1, 0, 1]
        validation_results = validate_model(predictions, ground_truth)
        st.write("Validation Results:")
        st.write(validation_results)

if __name__ == "__main__":
    main()