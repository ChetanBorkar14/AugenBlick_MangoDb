import streamlit as st
from components.charts import display_charts
from components.reports import generate_reports, extract_data_from_notebook
from components.tables import display_tables
from components.what_if import display_what_if_interface, load_dataset, display_dataset_info
from models.model_validation import validate_model
import pandas as pd
import os

# Fix file path handling
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "diabetes.csv")
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "main.ipynb")

def load_data(path: str) -> pd.DataFrame:
    """Load and validate data from CSV file"""
    try:
        data = pd.read_csv(path)
        required_cols = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="Diabetes Analysis App", page_icon="🏥", layout="wide")
    
    st.title("🏥 Diabetes Analysis and Prediction")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio(
        "Select a section:",
        ["Home", "Charts", "Reports", "Tables", "What-If Simulation", "Model Validation"]
    )
    
    if options == "Home":
        st.write("Welcome to the Diabetes Analysis and Prediction App!")
        st.markdown("""
        ### Available Sections:
        - **Charts**: Visual analysis of diabetes data
        - **Reports**: Detailed analysis reports
        - **Tables**: Summary statistics and data tables
        - **What-If Simulation**: Interactive prediction tool
        - **Model Validation**: Model performance metrics
        """)
    
    elif options == "Charts":
        st.subheader("📊 Data Visualizations")
        data = load_data(DATA_PATH)
        if not data.empty:
            display_charts(data)
    
    elif options == "Reports":
        st.subheader("📑 Analysis Reports")
        try:
            ns = extract_data_from_notebook(NOTEBOOK_PATH)
            generate_reports(ns)
        except FileNotFoundError:
            st.error(f"Notebook not found: {NOTEBOOK_PATH}")
    
    elif options == "Tables":
        st.subheader("📋 Summary Tables")
        try:
            ns = extract_data_from_notebook(NOTEBOOK_PATH)
            display_tables(ns)
        except Exception as e:
            st.error(f"Error displaying tables: {str(e)}")
    
    elif options == "What-If Simulation":
        st.subheader("🔮 What-If Scenario Simulation")
        
        # Allow both file upload and use of default dataset
        use_default = st.sidebar.checkbox("Use default dataset", value=True)
        
        if use_default:
            data = load_data(DATA_PATH)
        else:
            uploaded_file = st.file_uploader("Upload diabetes dataset (CSV)", type="csv")
            if uploaded_file is not None:
                data = load_dataset(uploaded_file)
            else:
                data = pd.DataFrame()
        
        if not data.empty:
            display_dataset_info(data)
            display_what_if_interface(data)
        else:
            st.info("Please upload a valid diabetes dataset or use the default dataset")
    
    elif options == "Model Validation":
        st.subheader("✅ Model Validation")
        data = load_data(DATA_PATH)
        if not data.empty:
            predictions = [0, 1, 0, 1, 0, 1, 0, 1]  # Replace with actual predictions
            ground_truth = data['Outcome'].tolist()[:8]  # Use actual outcomes
            validation_results = validate_model(predictions, ground_truth)
            st.write("Model Validation Results:")
            st.json(validation_results)

if __name__ == "__main__":
    main()