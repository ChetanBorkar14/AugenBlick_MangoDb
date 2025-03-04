import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load and prepare the diabetes dataset"""
    try:
        df = pd.read_csv(filepath)
        required_cols = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        
        # Verify required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Display dataset info
        st.sidebar.info(f"""
        📊 Dataset Statistics:
        - Total samples: {len(df)}
        - Average Glucose: {df['Glucose'].mean():.1f}
        - Diabetes cases: {df['Outcome'].sum()}
        - Non-diabetes cases: {len(df) - df['Outcome'].sum()}
        """)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

def display_dataset_info(data: pd.DataFrame):
    """Display detailed dataset statistics"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Analysis")
    
    stats = data.describe()
    metrics = ['Glucose', 'BMI', 'Age']
    
    for col in metrics:
        st.sidebar.write(f"**{col}:**")
        st.sidebar.write(f"- Mean: {stats[col]['mean']:.1f}")
        st.sidebar.write(f"- Range: {stats[col]['min']:.1f} - {stats[col]['max']:.1f}")

@st.cache_data
def train_model(data: pd.DataFrame,
               features: List[str],
               target_col: str = "Outcome") -> Tuple[Any, Any, Any]:
    """Train and cache prediction models"""
    try:
        # Scale features
        scaler = StandardScaler().fit(data[features])
        X = scaler.transform(data[features])
        y = data[target_col]
        
        # Train models
        model_logistic = LogisticRegression(random_state=42).fit(X, y)
        model_rf = RandomForestClassifier(random_state=42).fit(X, y)
        
        return scaler, model_logistic, model_rf
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        raise

def display_what_if_interface(data: pd.DataFrame):
    """Main interface for What-If scenario simulation"""
    st.title("🏥 Diabetes Risk What-If Analysis")
    
    if data.empty:
        st.error("Please provide a valid diabetes dataset")
        return

    # Define features for prediction
    features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    try:
        # Model settings
        st.sidebar.header("Model Settings")
        model_type = st.sidebar.selectbox(
            "Model Type",
            options=["logistic", "rf"],
            format_func=lambda x: "Logistic Regression" if x == "logistic" else "Random Forest",
            index=0
        )
        
        n_bootstrap = st.sidebar.number_input(
            "Bootstrap Iterations",
            min_value=100,
            max_value=2000,
            value=500,
            step=100
        )

        # Patient characteristics input
        st.subheader("Adjust Patient Characteristics")
        col1, col2 = st.columns(2)
        
        feature_values = {}
        for i, col in enumerate(features):
            with col1 if i % 2 == 0 else col2:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                default_val = float(data[col].mean())
                feature_values[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    format="%.1f"
                )

        # Prediction button
        if st.button("Predict Diabetes Risk", type="primary"):
            with st.spinner("Analyzing..."):
                scaler, model_logistic, model_rf = train_model(
                    data,
                    features
                )
                
                model = model_logistic if model_type == "logistic" else model_rf
                
                # Prepare prediction data
                X_pred = pd.DataFrame([feature_values])
                X_pred_scaled = scaler.transform(X_pred)
                
                # Make prediction
                predicted_prob = model.predict_proba(X_pred_scaled)[0, 1]
                st.success(f"Predicted Diabetes Risk: {predicted_prob:.1%}")
                
                # Feature importance
                if model_type == "rf":
                    importances = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.write("### Feature Importance")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=importances, x='Importance', y='Feature')
                    st.pyplot(fig)
                
                # Bootstrap analysis
                if n_bootstrap > 0:
                    with st.spinner("Calculating confidence intervals..."):
                        outcomes = []
                        progress_bar = st.progress(0)
                        
                        for i in range(n_bootstrap):
                            bootstrap_sample = data.sample(n=len(data), replace=True)
                            _, model_bs, _ = train_model(
                                bootstrap_sample,
                                features
                            )
                            pred = model_bs.predict_proba(X_pred_scaled)[0, 1]
                            outcomes.append(pred)
                            progress_bar.progress((i + 1) / n_bootstrap)
                        
                        outcomes = np.array(outcomes)
                        ci_lower, ci_upper = np.percentile(outcomes, [2.5, 97.5])
                        
                        st.write(f"95% Confidence Interval: ({ci_lower:.1%}, {ci_upper:.1%})")
                        
                        # Plot distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(outcomes, kde=True, ax=ax)
                        ax.set_title("Bootstrap Distribution of Predicted Risk")
                        ax.set_xlabel("Predicted Diabetes Risk")
                        ax.set_ylabel("Count")
                        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data format and try again")
        raise

if __name__ == "__main__":
    st.set_page_config(
        page_title="Diabetes Risk Prediction",
        page_icon="🏥",
        layout="wide"
    )
    
    st.markdown("""
    # 🏥 Diabetes Risk Prediction Tool
    This tool predicts diabetes risk based on patient characteristics.
    
    ### Required Dataset Columns:
    - Pregnancies
    - Glucose
    - BloodPressure
    - SkinThickness
    - Insulin
    - BMI
    - DiabetesPedigreeFunction
    - Age
    - Outcome (0 or 1)
    """)
    
    uploaded_file = st.file_uploader("Upload diabetes dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        data = load_dataset(uploaded_file)
        if not data.empty:
            display_dataset_info(data)
            display_what_if_interface(data)