<!DOCTYPE html>
<html>
<head>
    <title>Causal Inference on Diabetes Treatment</title>
</head>
<body>
    <h1>⚕️ Causal Inference on Diabetes Treatment</h1>
    <p>📊 Analyzing the causal effect of insulin treatment on diabetes diagnosis using the Indian KIMA Diabetes dataset.</p>
    
   <h2>📂 Dataset</h2>
    <p>The dataset consists of patient demographics, medical history, and treatment records.</p>
    <ul>
        <li><strong>💉 Treatment:</strong> Insulin (1 = Treated, 0 = Untreated)</li>
        <li><strong>📋 Outcome:</strong> Diabetes Diagnosis (1 = Yes, 0 = No)</li>
        <li><strong>📌 Confounders:</strong> Age, BMI, Blood Pressure, Pregnancies</li>
    </ul>
    
 <h2>⚙️ Methods</h2>
    <ul>
        <li><strong>🔗 Propensity Score Matching (PSM):</strong> Matches similar treated and untreated patients to balance confounders.</li>
        <li><strong>⚖️ Inverse Probability Weighting (IPW):</strong> Assigns weights based on treatment probability to simulate a randomized trial.</li>
        <li><strong>🛠️ Doubly Robust Estimation (AIPW):</strong> Combines IPW with outcome modeling for more reliable estimates.</li>
        <li><strong>🔍 What-If Scenario Simulation:</strong> Simulates different treatment assignments to evaluate their impact.</li>
        <li><strong>📈 Sensitivity Analysis:</strong> Assesses robustness against unmeasured confounders.</li>
    </ul>
    
 <h2>📊 Results</h2>
    <ul>
        <li><strong>📉 ATE:</strong> -0.0307 (3.07% reduction in diabetes probability)</li>
        <li><strong>✅ AIPW ATE:</strong> -0.0382 (More robust estimate)</li>
        <li><strong>📌 CATE:</strong></li>
        <ul>
            <li>👥 30-60: 0.0222</li>
            <li>🧒 0-30: -0.0625</li>
            <li>👴 60+: -0.3192</li>
        </ul>
    </ul>
    
 <h2>📊 Visualizations</h2>
    <ul>
        <li>📊 Propensity Score Distribution</li>
        <li>📉 Covariate Balance (Love Plots)</li>
        <li>📈 ATE & CATE Comparisons</li>
        <li>🔄 What-If Scenario Simulation</li>
    </ul>
    
 <h2>🚀 How to Run the Project</h2>
    <p>To install dependencies and run the analysis:</p>
    <pre>
    pip install pandas numpy matplotlib seaborn sklearn statsmodels dowhy
    python main.py
    </pre>
    <p>For the interactive dashboard (if implemented):</p>
    <pre>
    streamlit run app.py
    </pre>
    
 <h2>📌 Conclusion</h2>
    <p>📉 Our analysis suggests that insulin treatment slightly reduces diabetes probability, but the effect varies across age groups. Older patients (60+) see a stronger negative effect, while younger groups show mixed results. Further research and external validation are recommended.</p>
    
 <h2>🔮 Future Work</h2>
    <ul>
        <li>📊 Incorporate additional confounders like diet and physical activity.</li>
        <li>🤖 Apply machine learning models for better propensity score estimation.</li>
        <li>🔄 Expand What-If simulations with counterfactual analysis.</li>
    </ul>
    
   <h2>👥 Contributors</h2>
    <p>MangoDB-Aryan Bhagat and Chetan Borkar</p>
</body>
</html>
