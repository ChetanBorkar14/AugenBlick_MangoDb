import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def what_if_simulation(data: pd.DataFrame, modifications: dict) -> pd.DataFrame:
    """
    Returns a modified copy of 'data' with columns updated according to 'modifications'.
    """
    modified_data = data.copy()
    for parameter, value in modifications.items():
        # Try to convert the user input to the same dtype as original
        # (especially useful for numeric vs. string columns)
        try:
            modified_data[parameter] = modified_data[parameter].astype(type(value)).apply(lambda _: value)
        except:
            # If conversion fails, just set them as is (string fallback)
            modified_data[parameter] = str(value)
    return modified_data

def display_what_if_interface(data: pd.DataFrame):
    """
    Streamlit interface for a What-If scenario simulation:
      - Automatically includes all columns for user input
      - Generates distribution plots for numeric columns
      - Compares original vs. simulated data
    """
    st.title("🔍 What-If Scenario Simulation")
    st.markdown("""
    Adjust the values below to simulate how changes in each column might affect the dataset.
    Once you click **Run Simulation**, you'll see how these modifications change the data distribution 
    and summary statistics.
    """)

    # 1. Build 'modifications' dict for all columns
    modifications = {}
    st.write("### Adjust Parameters")
    for col in data.columns:
        # Numeric columns: use mean as default
        if pd.api.types.is_numeric_dtype(data[col]):
            default_value = float(data[col].mean())
            modifications[col] = st.number_input(f"Set new value for '{col}'", value=default_value)
        else:
            # Non-numeric columns: use the mode (most frequent value) as default
            default_text = str(data[col].mode().iloc[0]) if not data[col].mode().empty else ""
            modifications[col] = st.text_input(f"Set new value for '{col}'", value=default_text)

    # 2. Run the simulation
    if st.button("Run Simulation"):
        simulated_data = what_if_simulation(data, modifications)
        
        # 2A. Show simulated data preview
        st.subheader("📊 Simulated Data (Preview)")
        st.dataframe(simulated_data.head())
        
        # 2B. Display summary statistics for the simulated data
        st.subheader("📈 Summary Statistics (Simulated Data)")
        st.dataframe(simulated_data.describe().T)

        # 2C. Compare distributions for numeric columns
        st.subheader("📊 Distribution Comparison: Original vs. Simulated")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Create subplots for each numeric column
            ncols = 2  # 2 plots per row
            nrows = (len(numeric_cols) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4*nrows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, col in enumerate(numeric_cols):
                sns.histplot(data[col], kde=True, color='blue', label='Original', alpha=0.5, ax=axes[i])
                sns.histplot(simulated_data[col], kde=True, color='red', label='Simulated', alpha=0.5, ax=axes[i])
                axes[i].set_title(f"Distribution of '{col}'")
                axes[i].legend()

            # Hide extra axes if any
            for j in range(i+1, len(axes)):
                fig.delaxes(axes[j])

            st.pyplot(fig)
        else:
            st.warning("No numeric columns found for plotting distributions.")

        # 2D. Provide textual insights
        st.subheader("🧐 Insights & Conclusions")
        st.write("Below are the mean changes for each column you modified:")
        for parameter, value in modifications.items():
            if pd.api.types.is_numeric_dtype(data[parameter]):
                original_mean = data[parameter].mean()
                new_mean = simulated_data[parameter].mean()
                st.write(f"- **{parameter}**: changed to **{value}**. Original mean: {original_mean:.2f}, New mean: {new_mean:.2f}")
            else:
                st.write(f"- **{parameter}**: changed to **{value}** (non-numeric).")
        
        st.success("Simulation complete! Review the distributions, summary stats, and insights above.")
