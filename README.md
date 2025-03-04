# Diabetes Risk Prediction Tool 🏥

An interactive web application for predicting diabetes risk using machine learning models.

## Features

- 📉 Statistical Analysis using Graph
- 📋 Feature Importance Visualization
- 📊 Interactive What-If Analysis
- 🤖 Multiple ML Models (Logistic Regression, Random Forest)
- 📈 Bootstrap Confidence Intervals
- 🔄 Real-time Predictions

## Project Structure

```plaintext
AugenBlick_MangoDb/
├── notebooks/
│   └── main.ipynb
├── src/
│   ├── components/
│   │   ├── charts.py
│   │   ├── reports.py
│   │   ├── tables.py
│   │   └── what_if.py
│   ├── models/
│   │   └── model_validation.py
│   └── app.py
├── data/
│   └── diabetes.csv
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AugenBlick_MangoDb.git
cd AugenBlick_MangoDb

# Install dependencies
pip install -r requirements.txt
```

## Dataset used

## Running the Application

```bash
# Navigate to project directory
cd src

# Run the main application
streamlit run app.py

```

## Usage Guide

1. **Explore Features**

   - Navigate through different sections using the sidebar
   - View statistical summaries and visualizations

2. **What-If Analysis**

   - Adjust patient characteristics using the sliders
   - Choose between Logistic Regression or Random Forest models
   - Set bootstrap iterations for confidence intervals
   - Click "Predict" to see results

3. **Interpret Results**
   - View prediction probability
   - Check confidence intervals
   - Examine feature importance (Random Forest only)
   - Analyze bootstrap distribution

## Support

For support, please open an issue in the repository or contact the maintainers.

## 🤗 Happy Predicting!

- Remember to keep your dependencies updated
- Consider adding tests for new features
- Follow coding standards and documentation practices
