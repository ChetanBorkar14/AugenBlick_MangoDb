from datetime import datetime
import streamlit as st
import nbformat
from nbconvert import PythonExporter
import pandas as pd
import os
from PIL import Image

# Get the correct notebook path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "main.ipynb")

def extract_data_from_notebook(notebook_path):
    """
    Reads the notebook, converts it to a Python script, executes
    it in a local namespace, and returns key variables.
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    notebook = nbformat.reads(notebook_content, as_version=4)
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(notebook)
    
    local_ns = {}
    exec(source, local_ns)
    
    if 'df' not in local_ns:
        raise ValueError("Notebook does not define a DataFrame named 'df'.")
    
    return local_ns

def display_image(image_path, caption):
    """Displays an image if it exists."""
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=caption)
    else:
        st.warning(f"Image not found: {image_path}")

def format_number(value):
    """Formats numbers with color coding."""
    color = "green" if value > 0 else "red"
    return f'<span style="color:{color}; font-weight:bold;">{value:.4f}</span>'

def generate_reports(ns):
    """Generates the analysis report in Streamlit."""
    st.title("📊 Analysis Report")
    
    st.subheader("Data Overview")
    df = ns.get("df")
    st.write("Preview of the dataset:")
    st.dataframe(df.head())
    
    if "propensity_score" in df.columns:
        st.subheader("📈 Propensity Score Summary")
        st.table(df["propensity_score"].describe())
    else:
        st.warning("Propensity scores not found in the dataset.")
    
    st.subheader("📌 Key Analysis Results")
    
    if "ATE" in ns:
        ATE = ns["ATE"]
        st.markdown(f"**Average Treatment Effect (ATE):** {format_number(ATE)}", unsafe_allow_html=True)
    else:
        st.warning("ATE not computed in the notebook.")
        
    if "cate_results" in ns:
        st.write("**Conditional Average Treatment Effects (CATE) by Age Group:**")
        cate_results = ns["cate_results"]
        formatted_cate = {age_group: format_number(value) for age_group, value in cate_results.items()}
        st.markdown(pd.DataFrame.from_dict(formatted_cate, orient='index', columns=['CATE']).to_html(escape=False), unsafe_allow_html=True)
    else:
        st.warning("CATE not computed in the notebook.")
    
    if "aipw_effect" in ns:
        aipw_effect = ns["aipw_effect"]
        st.markdown(f"**Doubly-robust ATE (AIPW):** {format_number(aipw_effect)}", unsafe_allow_html=True)
    
    st.subheader("📊 Figures and Visualizations")
    for img, caption in [
        ("propensity_distribution.png", "Propensity Score Distribution"),
        ("balance_plot.png", "Covariate Balance Plot"),
        ("cate_by_age.png", "CATE by Age Group"),
        ("sensitivity_analysis.png", "Sensitivity Analysis")
    ]:
        display_image(os.path.join(BASE_DIR, "notebooks", img), caption)
    
    st.subheader("📝 Conclusion / Recommendation")
    if "conclusion" in ns:
        st.write(ns["conclusion"])
    elif "ATE" in ns:
        rec = "Exercise caution when prescribing insulin." if ns["ATE"] < 0 else "Consider insulin therapy for appropriate patients."
        st.write(f"**Conclusion:** {rec}")
    else:
        st.warning("No conclusion available from the notebook analysis.")
    
    st.subheader("📅 Report Generated On")
    st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    try:
        ns = extract_data_from_notebook(NOTEBOOK_PATH)
        generate_reports(ns)
    except Exception as e:
        st.error(str(e))