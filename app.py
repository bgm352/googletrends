import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np

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
    
    # Get numerical columns for simple statistics
    numerical_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if numerical_cols:
        # Simple statistics
        st.header("Summary Statistics")
        st.write(st.session_state.data[numerical_cols].describe())
        
        # Simple visualization with Streamlit's native charts
        st.header("Data Visualization")
        selected_col = st.selectbox("Select column for visualization", numerical_cols)
        
        if selected_col:
            chart_type = st.radio("Select chart type", ["Line Chart", "Bar Chart", "Area Chart"])
            
            if chart_type == "Line Chart":
                st.line_chart(st.session_state.data[selected_col])
            elif chart_type == "Bar Chart":
                st.bar_chart(st.session_state.data[selected_col])
            elif chart_type == "Area Chart":
                st.area_chart(st.session_state.data[selected_col])

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
