import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("Data Explorer App")

# Initialize session state for persistent data
if 'data' not in st.session_state:
    st.session_state.data = None
    
if 'filename' not in st.session_state:
    st.session_state.filename = None

# Sidebar for data upload and controls
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Attempt to read the file
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Excel file
                data = pd.read_excel(uploaded_file)
                
            st.session_state.data = data
            st.session_state.filename = uploaded_file.name
            st.success(f"Successfully loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Only show these options if data is loaded
    if st.session_state.data is not None:
        st.header("Analysis Options")
        
        # Get numerical columns for analysis
        numerical_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numerical_cols:
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Summary Statistics", "Data Visualization", "Correlation Analysis"]
            )
        else:
            st.warning("No numerical columns found for analysis")

# Main area - display data and analysis
if st.session_state.data is not None:
    st.header(f"Data Preview: {st.session_state.filename}")
    
    # Show data preview with pagination
    page_size = 10
    page_number = st.number_input('Page', min_value=1, value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    
    st.dataframe(st.session_state.data.iloc[start_idx:end_idx], use_container_width=True)
    
    # Data info
    with st.expander("Data Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Rows:** {st.session_state.data.shape[0]}")
            st.write(f"**Columns:** {st.session_state.data.shape[1]}")
        with col2:
            st.write("**Column Types:**")
            for col, dtype in zip(st.session_state.data.columns, st.session_state.data.dtypes):
                st.write(f"- {col}: {dtype}")
    
    # Analysis section based on sidebar selection
    if 'analysis_type' in locals():
        st.header(f"{analysis_type}")
        
        if analysis_type == "Summary Statistics":
            st.write(st.session_state.data.describe())
            
            # Check for missing values
            missing_values = st.session_state.data.isnull().sum()
            if missing_values.sum() > 0:
                st.subheader("Missing Values")
                st.write(missing_values[missing_values > 0])
        
        elif analysis_type == "Data Visualization":
            # Let user select columns to visualize
            selected_col = st.selectbox("Select column for visualization", numerical_cols)
            
            if selected_col:
                chart_type = st.radio("Select chart type", ["Histogram", "Box Plot", "Line Plot"])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if chart_type == "Histogram":
                    sns.histplot(data=st.session_state.data, x=selected_col, kde=True, ax=ax)
                    ax.set_title(f"Histogram of {selected_col}")
                
                elif chart_type == "Box Plot":
                    sns.boxplot(y=st.session_state.data[selected_col], ax=ax)
                    ax.set_title(f"Box Plot of {selected_col}")
                
                elif chart_type == "Line Plot":
                    if len(st.session_state.data) > 100:
                        # For large datasets, sample for better visualization
                        sample_data = st.session_state.data.sample(n=100)
                    else:
                        sample_data = st.session_state.data
                    
                    sns.lineplot(data=sample_data, y=selected_col, x=sample_data.index, ax=ax)
                    ax.set_title(f"Line Plot of {selected_col}")
                
                st.pyplot(fig)
        
        elif analysis_type == "Correlation Analysis":
            # Only include numerical columns
            corr_data = st.session_state.data[numerical_cols]
            
            # Calculate and display correlation matrix
            if len(numerical_cols) > 1:
                corr_matrix = corr_data.corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Correlation Matrix")
                st.pyplot(fig)
                
                # Show strongest correlations
                st.subheader("Top 5 Strongest Correlations")
                # Get the upper triangle of the correlation matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
                # Find the top 5 correlations
                highest_corr = upper.unstack().sort_values(ascending=False).head(5)
                for (col1, col2), corr_val in highest_corr.items():
                    st.write(f"**{col1}** and **{col2}**: {corr_val:.3f}")
            else:
                st.warning("Need at least two numerical columns for correlation analysis")

else:
    # Display instructions when no data is loaded
    st.info("👈 Please upload a data file using the sidebar to get started.")
    
    # Example section
    with st.expander("Example Data Sources"):
        st.markdown("""
        If you don't have a dataset handy, you can download sample datasets from these sources:
        - [Kaggle Datasets](https://www.kaggle.com/datasets)
        - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
        - [Data.gov](https://data.gov/)
        """)

# Footer
st.markdown("---")
st.markdown("Created with Streamlit ❤️")
