from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Dict, Union, Any
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def validate_model(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Union[str, int]]:
    """
    Validate model performance with multiple metrics
    
    Args:
        predictions: Model predictions
        ground_truth: True labels
        
    Returns:
        Dictionary containing validation metrics
    """
    try:
        # Convert inputs to numpy arrays
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall = recall_score(ground_truth, predictions, zero_division=0)
        f1 = f1_score(ground_truth, predictions, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        validation_results = {
            'Accuracy': f"{accuracy:.2%}",
            'Precision': f"{precision:.2%}",
            'Recall': f"{recall:.2%}",
            'F1 Score': f"{f1:.2%}",
            'True Negatives': int(tn),
            'False Positives': int(fp),
            'False Negatives': int(fn),
            'True Positives': int(tp)
        }

        return validation_results

    except Exception as e:
        st.error(f"Validation error: {str(e)}")
        return {}

def analyze_fairness(data: pd.DataFrame, 
                    predictions: np.ndarray, 
                    sensitive_features: List[str]) -> Dict[str, Any]:
    """
    Analyze model fairness across sensitive features
    
    Args:
        data: Input DataFrame
        predictions: Model predictions
        sensitive_features: List of features to analyze for fairness
        
    Returns:
        Dictionary containing fairness metrics for each feature
    """
    try:
        results = {}
        
        for feature in sensitive_features:
            if feature not in data.columns:
                st.warning(f"Feature '{feature}' not found in dataset")
                continue
                
            # Calculate metrics for each group
            group_metrics = {}
            for value in sorted(data[feature].unique()):
                mask = data[feature] == value
                if sum(mask) > 0:  # Only if group has samples
                    group_pred = predictions[mask]
                    group_true = data.loc[mask, 'Outcome']
                    
                    group_metrics[f'Group {value}'] = {
                        'Size': sum(mask),
                        'Positive Rate': f"{np.mean(group_pred):.2%}",
                        'Accuracy': f"{accuracy_score(group_true, group_pred):.2%}"
                    }
            
            results[feature] = group_metrics
            
        return results
        
    except Exception as e:
        st.error(f"Fairness analysis error: {str(e)}")
        return {}

def display_model_validation(data: pd.DataFrame, 
                           model: Any, 
                           features: List[str],
                           test_size: float = 0.2) -> None:
    """
    Display model validation results in Streamlit
    
    Args:
        data: Input DataFrame
        model: Trained model object
        features: List of feature names
        test_size: Proportion of test set
    """
    try:
        # Make predictions
        X = data[features]
        y_true = data['Outcome']
        y_pred = model.predict(X)
        
        # Get validation metrics
        metrics = validate_model(y_pred, y_true)
        
        # Display metrics
        st.subheader("Model Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", metrics['Accuracy'])
            st.metric("Precision", metrics['Precision'])
        
        with col2:
            st.metric("Recall", metrics['Recall'])
            st.metric("F1 Score", metrics['F1 Score'])
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        cm_data = {
            'Predicted Negative': [metrics['True Negatives'], metrics['False Negatives']],
            'Predicted Positive': [metrics['False Positives'], metrics['True Positives']]
        }
        cm_df = pd.DataFrame(cm_data, index=['Actual Negative', 'Actual Positive'])
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title('Feature Importance')
            st.pyplot(fig)
        
        # Analyze fairness
        sensitive_features = ['Age', 'Pregnancies']  # Add relevant features
        fairness_results = analyze_fairness(data, y_pred, sensitive_features)
        
        if fairness_results:
            st.subheader("Fairness Analysis")
            for feature, metrics in fairness_results.items():
                st.write(f"**{feature} Analysis:**")
                metrics_df = pd.DataFrame(metrics).T
                st.table(metrics_df)
                
    except Exception as e:
        st.error(f"Error in model validation display: {str(e)}")
        st.error("Please check your data and model configuration")