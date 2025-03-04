from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def validate_model(predictions, ground_truth):
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)

    validation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return validation_results

def check_bias(predictions, ground_truth, sensitive_features):
    results = {}
    
    # Convert predictions and ground_truth to a DataFrame if they are not already
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)
    if not isinstance(ground_truth, pd.Series):
        ground_truth = pd.Series(ground_truth)

    # Add ground truth to predictions DataFrame
    predictions['ground_truth'] = ground_truth

    for feature in sensitive_features:
        group_0 = predictions[predictions[feature] == 0]['ground_truth']
        group_1 = predictions[predictions[feature] == 1]['ground_truth']

        results[feature] = {
            'group_0_mean': group_0.mean(),
            'group_1_mean': group_1.mean(),
            'difference': group_1.mean() - group_0.mean()
        }

    return pd.DataFrame(results)

# Example usage:
# predictions = pd.Series([0, 1, 1, 0, 1])
# ground_truth = pd.Series([0, 1, 0, 0, 1])
# sensitive_features = ['sensitive_feature_1', 'sensitive_feature_2']
# bias_results = check_bias(predictions, ground_truth, sensitive_features)