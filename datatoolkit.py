import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit_shadcn_ui as ui

st.set_page_config(page_title="Data Analysis App", layout="wide")
st.title("Data Analysis & EDA Dashboard")

# -----------------------------
# Demo / Upload Dataset
# -----------------------------
use_demo = st.checkbox("Use demo dataset")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

if use_demo:
    df = sns.load_dataset("penguins")
    st.success(f"Demo dataset loaded!")
elif uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"File loaded successfully! Shape: {df.shape}")
else:
    st.info("Please upload a file or use the demo dataset to get started.")
    st.stop()

# -----------------------------
# Initialize session state
# -----------------------------
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = df.copy()

# -----------------------------
# KPI / Summary Panel
# -----------------------------
kpi_container = st.container()
with kpi_container:
    col1, col2, col3, col4 = st.columns(4)
    rows_metric = col1.empty()
    cols_metric = col2.empty()
    missing_metric = col3.empty()
    duplicates_metric = col4.empty()

def update_kpis():
    df_current = st.session_state.df_cleaned
    rows_metric.metric("Rows", df_current.shape[0])
    cols_metric.metric("Columns", df_current.shape[1])
    missing_metric.metric("Total Missing Values", df_current.isnull().sum().sum())
    duplicates_metric.metric("Duplicate Rows", df_current.duplicated().sum())

update_kpis()

# -----------------------------
# Tabs Layout
# -----------------------------
tab_preview, tab_cleaning, tab_stats, tab_sql = st.tabs(
    ["Overview & Info", "Data Cleaning", "Statistics & Visuals", "SQL / Download"]
)

# -----------------------------
# Tab 1: Preview & Info
# -----------------------------
with tab_preview:
    st.subheader("Dataset Overview")
    show_full = st.checkbox("Show entire dataset", value=False, key="show_full")
    st.dataframe(st.session_state.df_cleaned if show_full else st.session_state.df_cleaned.head())

    st.subheader("Column Overview")
    # Always recreate from current df_cleaned in session_state
    df_current = st.session_state.df_cleaned
    col_overview = pd.DataFrame({
        "Column": df_current.columns,
        "Data Type": df_current.dtypes.astype(str),
        "Missing Count": df_current.isnull().sum(),
        "Missing %": (df_current.isnull().mean() * 100).round(2)
    }).sort_values("Missing %", ascending=False)
    st.table(col_overview)

    if st.checkbox("Show missing values heatmap", key="heatmap_preview"):
        plt.figure(figsize=(10,6))
        sns.heatmap(df_current.isnull(), cbar=False, cmap='viridis')
        st.pyplot(plt.gcf())
        plt.clf()


# -----------------------------
# Tab 2: Data Cleaning
# -----------------------------
with tab_cleaning:
    st.subheader("Handle Missing Values")
    missing_cols = st.session_state.df_cleaned.columns[st.session_state.df_cleaned.isnull().any()].tolist()
    if missing_cols:
        col_to_clean = st.selectbox("Select column to clean", missing_cols)
        strategy = st.selectbox("Imputation strategy", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"])
        if strategy == "Fill with custom value":
            custom_value = st.text_input("Custom fill value")
        if st.button("Apply missing value handling", key="apply_missing"):
            if strategy == "Drop rows":
                st.session_state.df_cleaned.dropna(subset=[col_to_clean], inplace=True)
            elif strategy == "Fill with mean":
                st.session_state.df_cleaned[col_to_clean].fillna(st.session_state.df_cleaned[col_to_clean].mean(), inplace=True)
            elif strategy == "Fill with median":
                st.session_state.df_cleaned[col_to_clean].fillna(st.session_state.df_cleaned[col_to_clean].median(), inplace=True)
            elif strategy == "Fill with mode":
                st.session_state.df_cleaned[col_to_clean].fillna(st.session_state.df_cleaned[col_to_clean].mode()[0], inplace=True)
            elif strategy == "Fill with custom value":
                st.session_state.df_cleaned[col_to_clean].fillna(custom_value, inplace=True)
            st.success(f"Missing values handled for {col_to_clean}. New shape: {st.session_state.df_cleaned.shape}")
            st.dataframe(st.session_state.df_cleaned.head())
            update_kpis()
    else:
        st.info("No columns with missing values found.")

    st.subheader("Drop Columns")
    cols_to_drop = st.multiselect("Select columns to drop", st.session_state.df_cleaned.columns)
    if st.button("Drop selected columns", key="drop_cols"):
        st.session_state.df_cleaned.drop(columns=cols_to_drop, inplace=True)
        st.success(f"Dropped columns: {cols_to_drop}. New shape: {st.session_state.df_cleaned.shape}")
        st.dataframe(st.session_state.df_cleaned.head())
        update_kpis()

    st.subheader("Handle Duplicates")
    if st.button("Remove duplicate rows", key="remove_dupes"):
        before = st.session_state.df_cleaned.shape[0]
        st.session_state.df_cleaned.drop_duplicates(inplace=True)
        after = st.session_state.df_cleaned.shape[0]
        st.success(f"Removed {before-after} duplicate rows. New shape: {st.session_state.df_cleaned.shape}")
        update_kpis()

# -----------------------------
# Tab 3: Statistics & Visuals
# -----------------------------
with tab_stats:
    numeric_cols = st.session_state.df_cleaned.select_dtypes(include='number').columns.tolist()
    cat_cols = st.session_state.df_cleaned.select_dtypes(include=['object','category']).columns.tolist()

    st.subheader("Descriptive Statistics")
    if numeric_cols:
        st.write(st.session_state.df_cleaned[numeric_cols].describe().T.assign(
            skew=st.session_state.df_cleaned[numeric_cols].skew(),
            kurt=st.session_state.df_cleaned[numeric_cols].kurtosis()
        ))
    else:
        st.info("No numeric columns for descriptive stats.")

    if cat_cols:
        st.write(st.session_state.df_cleaned[cat_cols].describe().T)
    else:
        st.info("No categorical columns found.")

    st.subheader("Correlation Analysis")
    if len(numeric_cols) >= 2:
        corr_matrix = st.session_state.df_cleaned[numeric_cols].corr()
        st.dataframe(corr_matrix)
        if st.checkbox("Show correlation heatmap", key="heatmap_corr"):
            plt.figure(figsize=(10,6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt.gcf())
            plt.clf()
    else:
        st.info("Not enough numeric columns for correlation analysis.")

    st.subheader("Visualizations")
    if numeric_cols:
        col1, col2 = st.columns(2)
        col_hist = col1.selectbox("Select numeric column for histogram", numeric_cols)
        bins = col1.slider("Number of bins", min_value=5, max_value=100, value=30)
        plt.figure(figsize=(6,4))
        sns.histplot(st.session_state.df_cleaned[col_hist], bins=bins, kde=True)
        st.pyplot(plt.gcf())
        plt.clf()

        col_box = col2.selectbox("Select numeric column for box plot", numeric_cols, key="box_tab")
        plt.figure(figsize=(6,4))
        sns.boxplot(x=st.session_state.df_cleaned[col_box])
        st.pyplot(plt.gcf())
        plt.clf()

        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis for scatter plot", numeric_cols, key="scatter_x_tab")
            y_col = st.selectbox("Y-axis for scatter plot", numeric_cols, key="scatter_y_tab")
            plt.figure(figsize=(6,4))
            sns.scatterplot(x=st.session_state.df_cleaned[x_col], y=st.session_state.df_cleaned[y_col])
            st.pyplot(plt.gcf())
            plt.clf()
    else:
        st.info("No numeric columns for visualizations.")

# -----------------------------
# Tab 4: SQL Sandbox & Download
# -----------------------------
with tab_sql:
    st.subheader("SQL Sandbox")
    conn = sqlite3.connect(":memory:")
    st.session_state.df_cleaned.to_sql("df_cleaned", conn, index=False, if_exists="replace")

    sample_columns = st.session_state.df_cleaned.columns[:3].tolist()
    if len(sample_columns) < 2:
        example_query = f"SELECT {', '.join(sample_columns)} FROM df_cleaned;"
    else:
        example_query = f"SELECT {sample_columns[0]}, {sample_columns[1]} FROM df_cleaned;"
    st.write(f"Example query using your columns:\n`{example_query}`")

    query = st.text_area("Enter SQL query here:", height=100, value=example_query)
    if st.button("Run Query", key="run_sql"):
        if query.strip():
            try:
                result = pd.read_sql_query(query, conn)
                st.success(f"Query executed! {result.shape[0]} rows returned.")
                st.dataframe(result)
            except Exception as e:
                st.error(f"Error in query: {e}")
        else:
            st.info("Please enter a SQL query.")

    st.subheader("Download Cleaned Dataset")
    csv_data = st.session_state.df_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Final Dataset",
        data=csv_data,
        file_name="final_dataset.csv",
        mime="text/csv"
    )
