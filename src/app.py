import streamlit as st
from components.charts import display_charts
from components.reports import generate_reports, extract_data_from_notebook
from components.tables import display_tables
from components.what_if import display_what_if_interface, load_dataset, display_dataset_info
from models.model_validation import validate_model, display_model_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# Fix file path handling
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "diabetes.csv")
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "main.ipynb")

def load_data(path: str) -> pd.DataFrame:
    """Load and validate diabetes dataset"""
    try:
        data = pd.read_csv(path)
        required_cols = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        
        # Validate columns
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
            
        # Check data quality
        if data.isnull().any().any():
            st.warning("Dataset contains missing values")
            # Fill missing values with median
            data = data.fillna(data.median())
            
        # Validate outcome values
        if not all(data['Outcome'].isin([0, 1])):
            st.error("Outcome column must contain only binary values (0 or 1)")
            return pd.DataFrame()
            
        # Display basic statistics
        st.sidebar.info(f"""
        📊 Dataset Info:
        - Total samples: {len(data)}
        - Diabetes cases: {data['Outcome'].sum()}
        - Non-diabetes cases: {len(data) - data['Outcome'].sum()}
        """)
            
        return data
        
    except FileNotFoundError:
        st.error(f"Dataset not found at: {path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="Diabetes Analysis App", page_icon="🏥", layout="wide")
    
    st.title("🏥 Diabetes Analysis and Prediction")
    
    # Sidebar navigation
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
        
        ### Dataset Information
        The dataset contains various health parameters used to predict diabetes:
        - Pregnancies
        - Glucose levels
        - Blood Pressure
        - Skin Thickness
        - Insulin levels
        - BMI (Body Mass Index)
        - Diabetes Pedigree Function
        - Age
        - Outcome (0 = No Diabetes, 1 = Diabetes)
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
        try:
            data = load_data(DATA_PATH)
            if not data.empty:
                features = [
                    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
                ]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    model_type = st.selectbox(
                        "Select Model",
                        options=["Logistic Regression", "Random Forest"],
                        index=0
                    )
                
                with col2:
                    test_size = st.slider(
                        "Test Set Size (%)", 
                        min_value=10, 
                        max_value=40, 
                        value=20
                    )
                
                # Prepare data
                X = data[features]
                y = data['Outcome']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )
                
                # Train model
                with st.spinner("Training model..."):
                    if model_type == "Logistic Regression":
                        model = LogisticRegression(random_state=42)
                    else:
                        model = RandomForestClassifier(
                            n_estimators=100,
                            random_state=42
                        )
                    
                    model.fit(X_train, y_train)
                    
                    # Display validation results
                    display_model_validation(
                        data=data,
                        model=model,
                        features=features,
                        test_size=test_size/100
                    )
            else:
                st.error("Please ensure the diabetes dataset is available")
                
        except Exception as e:
            st.error(f"Error in model validation: {str(e)}")
            st.error("Please check your data and try again")

if __name__ == "__main__":
    main()