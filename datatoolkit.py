# streamlit_data_app_robust.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Analysis App", layout="wide")
st.title("Data Analysis & EDA Dashboard")

# -----------------------------
# Demo dataset option
# -----------------------------
use_demo = st.checkbox("Use demo dataset")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

if use_demo:
    # Load a demo dataset
    df = sns.load_dataset("penguins")  # You can replace with any other demo dataset
    st.success(f"Demo dataset loaded! Shape: {df.shape}")
    df_cleaned = df.copy()

elif uploaded_file:
    # Load uploaded dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"File loaded successfully! Shape: {df.shape}")
    df_cleaned = df.copy()

else:
    st.info("Please upload a file or use the demo dataset to get started.")
    st.stop()

# -----------------------------
# Dataset Preview
# -----------------------------
with st.expander("Dataset Preview", expanded=True):
    show_full = st.checkbox("Show entire dataset", value=False)
    st.dataframe(df_cleaned if show_full else df_cleaned.head())

# -----------------------------
# Column Data Types
# -----------------------------
with st.expander("Column Data Types", expanded=False):
    dtype_df = pd.DataFrame({
        "Column": df_cleaned.columns,
        "Data Type": df_cleaned.dtypes.astype(str)
    })
    st.table(dtype_df)

# -----------------------------
# Missing Values
# -----------------------------
with st.expander("Missing Values Summary", expanded=False):
    missing_df = pd.DataFrame({
        "Missing Count": df_cleaned.isnull().sum(),
        "Missing %": df_cleaned.isnull().mean() * 100
    }).sort_values("Missing %", ascending=False)
    st.write(missing_df)

    if st.checkbox("Show missing values heatmap"):
        plt.figure(figsize=(10,6))
        sns.heatmap(df_cleaned.isnull(), cbar=False, cmap='viridis')
        st.pyplot(plt.gcf())
        plt.clf()

# -----------------------------
# Data Cleaning
# -----------------------------
with st.expander("Data Cleaning", expanded=False):
    st.subheader("Handle Missing Values")
    missing_cols = df_cleaned.columns[df_cleaned.isnull().any()].tolist()
    if missing_cols:
        col_to_clean = st.selectbox("Select column to clean", missing_cols)
        strategy = st.selectbox("Imputation strategy", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"])
        if strategy == "Fill with custom value":
            custom_value = st.text_input("Custom fill value")
        if st.button("Apply missing value handling"):
            if strategy == "Drop rows":
                df_cleaned.dropna(subset=[col_to_clean], inplace=True)
            elif strategy == "Fill with mean":
                df_cleaned[col_to_clean].fillna(df_cleaned[col_to_clean].mean(), inplace=True)
            elif strategy == "Fill with median":
                df_cleaned[col_to_clean].fillna(df_cleaned[col_to_clean].median(), inplace=True)
            elif strategy == "Fill with mode":
                df_cleaned[col_to_clean].fillna(df_cleaned[col_to_clean].mode()[0], inplace=True)
            elif strategy == "Fill with custom value":
                df_cleaned[col_to_clean].fillna(custom_value, inplace=True)
            st.success(f"Missing values handled for {col_to_clean}. New shape: {df_cleaned.shape}")
            st.dataframe(df_cleaned.head())
    else:
        st.info("No columns with missing values found.")

    st.subheader("Drop Columns")
    cols_to_drop = st.multiselect("Select columns to drop", df_cleaned.columns)
    if st.button("Drop selected columns"):
        df_cleaned.drop(columns=cols_to_drop, inplace=True)
        st.success(f"Dropped columns: {cols_to_drop}. New shape: {df_cleaned.shape}")
        st.dataframe(df_cleaned.head())

    st.subheader("Handle Duplicates")
    if st.button("Remove duplicate rows"):
        before = df_cleaned.shape[0]
        df_cleaned.drop_duplicates(inplace=True)
        after = df_cleaned.shape[0]
        st.success(f"Removed {before-after} duplicate rows. New shape: {df_cleaned.shape}")

# -----------------------------
# Descriptive Statistics
# -----------------------------
with st.expander("Descriptive Statistics", expanded=False):
    numeric_cols = df_cleaned.select_dtypes(include='number').columns.tolist()
    cat_cols = df_cleaned.select_dtypes(include=['object','category']).columns.tolist()

    st.subheader("Numeric Columns Summary")
    if numeric_cols:
        st.write(df_cleaned[numeric_cols].describe().T.assign(
            skew=df_cleaned[numeric_cols].skew(), kurt=df_cleaned[numeric_cols].kurtosis()
        ))
    else:
        st.info("No numeric columns for descriptive stats.")

    st.subheader("Categorical Columns Summary")
    if cat_cols:
        st.write(df_cleaned[cat_cols].describe().T)
    else:
        st.info("No categorical columns found.")

    # Low variance numeric columns
    if numeric_cols:
        low_var_cols = df_cleaned[numeric_cols].std()
        low_var_cols = low_var_cols[low_var_cols < 1e-3].index.tolist()
        if low_var_cols:
            st.warning(f"Low variance numeric columns: {low_var_cols}")

# -----------------------------
# Correlation Analysis
# -----------------------------
with st.expander("Correlation Analysis", expanded=False):
    if len(numeric_cols) >= 2:
        corr_matrix = df_cleaned[numeric_cols].corr()
        st.write("Correlation matrix:")
        st.dataframe(corr_matrix)

        if st.checkbox("Show correlation heatmap"):
            plt.figure(figsize=(10,6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt.gcf())
            plt.clf()
    else:
        st.info("Not enough numeric columns for correlation analysis.")

# -----------------------------
# Interactive Visualizations
# -----------------------------
with st.expander("Interactive Visualizations", expanded=False):
    if numeric_cols:
        st.subheader("Histogram / Distribution")
        col_hist = st.selectbox("Select numeric column for histogram", numeric_cols)
        bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)
        plt.figure(figsize=(8,4))
        sns.histplot(df_cleaned[col_hist], bins=bins, kde=True)
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("Box Plot for Outliers")
        col_box = st.selectbox("Select numeric column for box plot", numeric_cols, key="box")
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df_cleaned[col_box])
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("Scatter Plot")
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
            plt.figure(figsize=(6,4))
            sns.scatterplot(x=df_cleaned[x_col], y=df_cleaned[y_col])
            st.pyplot(plt.gcf())
            plt.clf()
    else:
        st.info("No numeric columns for visualizations.")

# -----------------------------
# Grouping & Aggregation
# -----------------------------
with st.expander("Grouping & Aggregation", expanded=False):
    if cat_cols and numeric_cols:
        group_col = st.selectbox("Group by categorical column", cat_cols, key="groupby")
        agg_col = st.selectbox("Aggregate numeric column", numeric_cols, key="agg")
        agg_func = st.selectbox("Aggregation function", ["sum", "mean", "median", "max", "min"])
        if st.button("Compute aggregation"):
            grouped = df_cleaned.groupby(group_col)[agg_col].agg(agg_func)
            st.dataframe(grouped)

# -----------------------------
# Outlier Detection (interactive)
# -----------------------------
with st.expander("Outlier Detection", expanded=False):
    if numeric_cols:
        outlier_cols = st.multiselect("Choose numeric columns to check for outliers", numeric_cols, key="outlier_cols")
        if outlier_cols:
            method = st.selectbox("Detection Method", ("IQR", "Z-score"), key="outlier_method")
            df_outliers = df_cleaned.copy()
            outlier_indices = set()

            if method == "IQR":
                multiplier = st.slider("IQR multiplier", min_value=1.0, max_value=3.0, value=1.5)
                for col in outlier_cols:
                    Q1 = df_outliers[col].quantile(0.25)
                    Q3 = df_outliers[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - multiplier * IQR
                    upper = Q3 + multiplier * IQR
                    outlier_indices.update(df_outliers[(df_outliers[col] < lower) | (df_outliers[col] > upper)].index)
            elif method == "Z-score":
                threshold = st.slider("Z-score threshold", min_value=2.0, max_value=5.0, value=3.0)
                for col in outlier_cols:
                    z_scores = (df_outliers[col] - df_outliers[col].mean()) / df_outliers[col].std()
                    outlier_indices.update(df_outliers[np.abs(z_scores) > threshold].index)

            st.write(f"Number of outlier rows detected: {len(outlier_indices)}")
            if len(outlier_indices) > 0:
                st.dataframe(df_outliers.loc[list(outlier_indices)])
                if st.checkbox("Remove detected outliers"):
                    df_cleaned = df_cleaned.drop(index=outlier_indices)
                    st.success(f"Outliers removed. New dataset shape: {df_cleaned.shape}")
                    st.dataframe(df_cleaned.head())
    else:
        st.info("No numeric columns for outlier detection.")

# -----------------------------
# SQL Sandbox (SQLite with dynamic example)
# -----------------------------
with st.expander("SQL Sandbox", expanded=False):
    st.subheader("Run SQL Queries on Cleaned Dataset")
    conn = sqlite3.connect(":memory:")
    df_cleaned.to_sql("df_cleaned", conn, index=False, if_exists="replace")

    sample_columns = df_cleaned.columns[:3].tolist()
    if len(sample_columns) < 2:
        example_query = f"SELECT {', '.join(sample_columns)} FROM df_cleaned;"
    else:
        example_query = f"SELECT {sample_columns[0]}, {sample_columns[1]} FROM df_cleaned;"
    st.write(f"Example query using your columns:\n`{example_query}`")

    query = st.text_area("Enter SQL query here:", height=100, value=example_query)
    if st.button("Run Query"):
        if query.strip():
            try:
                result = pd.read_sql_query(query, conn)
                st.success(f"Query executed! {result.shape[0]} rows returned.")
                st.dataframe(result)
            except Exception as e:
                st.error(f"Error in query: {e}")
        else:
            st.info("Please enter a SQL query.")

# -----------------------------
# Final Download
# -----------------------------
csv_data = df_cleaned.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download Final Dataset",
    data=csv_data,
    file_name="final_dataset.csv",
    mime="text/csv"
)
