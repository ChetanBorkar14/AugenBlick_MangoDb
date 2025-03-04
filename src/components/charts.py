import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_histogram(data, column, bins=30, color='skyblue'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=bins, kde=True, color=color)
    plt.title(f'Histogram of {column}', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()  # Clear plot after rendering

def plot_scatter(data, x_column, y_column, color='green'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_column, y=y_column, color=color)
    plt.title(f'Scatter plot of {x_column} vs {y_column}', fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

def plot_boxplot(data, x_column, y_column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x_column, y=y_column, palette='Set2')
    plt.title(f'Boxplot of {y_column} by {x_column}', fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

def plot_line_chart(data, x_column, y_column, color='purple'):
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_column], data[y_column], marker='o', linestyle='-', color=color)
    plt.title(f'Line chart of {y_column} over {x_column}', fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

def display_charts(data):
    st.title("📊 Data Visualizations")
    st.write("Select the type of chart and columns to visualize the dataset.")

    # Sidebar for user selections
    st.sidebar.header("Chart Settings")
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Histogram", "Scatter Plot", "Box Plot", "Line Chart"])

    if chart_type == "Histogram":
        column = st.sidebar.selectbox("Select Column", data.columns)
        bins = st.sidebar.slider("Number of Bins", min_value=10, max_value=100, value=30)
        color = st.sidebar.color_picker("Pick a color", "#87CEEB")  # Default skyblue
        plot_histogram(data, column, bins, color)

    elif chart_type == "Scatter Plot":
        x_column = st.sidebar.selectbox("Select X Column", data.columns)
        y_column = st.sidebar.selectbox("Select Y Column", data.columns)
        color = st.sidebar.color_picker("Pick a color", "#32CD32")  # Default green
        if x_column != y_column:
            plot_scatter(data, x_column, y_column, color)
        else:
            st.warning("⚠️ X and Y columns should be different!")

    elif chart_type == "Box Plot":
        x_column = st.sidebar.selectbox("Select X Column (Categorical)", data.columns)
        y_column = st.sidebar.selectbox("Select Y Column (Numerical)", data.columns)
        if x_column != y_column:
            plot_boxplot(data, x_column, y_column)
        else:
            st.warning("⚠️ X and Y columns should be different!")

    elif chart_type == "Line Chart":
        x_column = st.sidebar.selectbox("Select X Column", data.columns)
        y_column = st.sidebar.selectbox("Select Y Column", data.columns)
        color = st.sidebar.color_picker("Pick a color", "#800080")  # Default purple
        if x_column != y_column:
            plot_line_chart(data, x_column, y_column, color)
        else:
            st.warning("⚠️ X and Y columns should be different!")
