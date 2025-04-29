import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import time
import json
import os
import sqlite3
import threading
import schedule
from sqlalchemy import create_engine, text
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Healthcare Search Trends Analyzer",
    page_icon="🏥",
    layout="wide",
)

# Database initialization
def init_db():
    """Initialize SQLite database for storing trends data"""
    try:
        engine = create_engine('sqlite:///healthcare_trends.db')
        # Create tables
        with engine.connect() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS trends_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT,
                    region TEXT,
                    state TEXT,
                    interest FLOAT,
                    timestamp DATETIME
                )
            '''))
            
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS demographic_data (
                    state TEXT PRIMARY KEY,
                    population INTEGER,
                    median_age FLOAT,
                    income_per_capita INTEGER,
                    healthcare_coverage_pct FLOAT,
                    last_updated DATETIME
                )
            '''))
            
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS tracked_terms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT UNIQUE,
                    category TEXT,
                    added_date DATETIME,
                    last_updated DATETIME
                )
            '''))
        logger.info("Database initialized successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        st.error(f"Database initialization error: {str(e)}")
        return None

# Initialize PyTrends client with rate limiting
@st.cache_resource
def get_pytrends_client():
    """Create and return a PyTrends client"""
    return TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)

# Function to get Google Trends data with retry logic
def get_google_trends_data_with_retry(keywords, geo, timeframe='today 12-m', max_retries=3):
    """Get Google Trends data with retry logic for rate limiting"""
    retries = 0
    while retries < max_retries:
        try:
            pytrends = get_pytrends_client()
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
            data = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True)
            logger.info(f"Successfully retrieved trends data for {keywords} in {geo}")
            return data
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                retries += 1
                wait_time = 60 * retries  # Increase wait time with each retry
                logger.warning(f"Rate limited by Google. Retry {retries}/{max_retries} in {wait_time} seconds...")
                time.sleep(wait_time)  # Exponential backoff
            else:
                logger.error(f"Error fetching Google Trends data: {str(e)}")
                st.error(f"Error fetching Google Trends data: {str(e)}")
                return pd.DataFrame()
    
    logger.error("Failed to retrieve data after multiple attempts")
    st.error("Failed to retrieve data after multiple attempts. Try again later.")
    return pd.DataFrame()

# Store trends data in database
def store_trends_data(term, region, data):
    """Store Google Trends data in the database"""
    try:
        engine = init_db()
        if engine is None:
            return False
            
        timestamp = datetime.now()
        
        # Convert DataFrame to records for insertion
        if not data.empty:
            data = data.reset_index()
            records = []
            for _, row in data.iterrows():
                records.append({
                    'term': term,
                    'region': region,
                    'state': row['geoName'],
                    'interest': float(row[term]),
                    'timestamp': timestamp
                })
            
            # Insert into database
            df = pd.DataFrame(records)
            df.to_sql('trends_data', engine, if_exists='append', index=False)
            
            # Update tracked terms
            with engine.connect() as conn:
                conn.execute(text('''
                    INSERT INTO tracked_terms (term, category, added_date, last_updated)
                    VALUES (:term, :category, :added_date, :last_updated)
                    ON CONFLICT(term) DO UPDATE SET
                    last_updated = excluded.last_updated
                '''), {
                    'term': term,
                    'category': 'custom',
                    'added_date': timestamp,
                    'last_updated': timestamp
                })
            
            logger.info(f"Stored trends data for {term} in {region}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error storing trends data: {str(e)}")
        return False

# Load demographic data from database
def load_demographic_data():
    """Load demographic data for US states from database"""
    try:
        engine = init_db()
        if engine is None:
            return pd.DataFrame()
            
        query = "SELECT * FROM demographic_data"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Error loading demographic data: {str(e)}")
        return pd.DataFrame()

# Load trends data from database
def load_trends_data(term=None, region=None, start_date=None, end_date=None):
    """Load trends data from database with optional filters"""
    try:
        engine = init_db()
        if engine is None:
            return pd.DataFrame()
            
        query = "SELECT * FROM trends_data WHERE 1=1"
        params = {}
        
        if term:
            query += " AND term = :term"
            params['term'] = term
            
        if region:
            query += " AND region = :region"
            params['region'] = region
            
        if start_date:
            query += " AND timestamp >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND timestamp <= :end_date"
            params['end_date'] = end_date
            
        df = pd.read_sql(query, engine, params=params)
        return df
    except Exception as e:
        logger.error(f"Error loading trends data: {str(e)}")
        return pd.DataFrame()

# Get list of tracked terms
def get_tracked_terms():
    """Get list of all tracked search terms"""
    try:
        engine = init_db()
        if engine is None:
            return []
            
        query = "SELECT term, category, added_date, last_updated FROM tracked_terms ORDER BY last_updated DESC"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Error getting tracked terms: {str(e)}")
        return pd.DataFrame()

# Scheduled data collection job
def scheduled_data_collection():
    """Function to run scheduled data collection jobs"""
    try:
        terms = get_tracked_terms()
        if terms.empty:
            logger.info("No terms to track. Skipping scheduled collection.")
            return
            
        for _, row in terms.iterrows():
            term = row['term']
            data = get_google_trends_data_with_retry([term], geo='US')
            if not data.empty:
                store_trends_data(term, 'US', data)
                logger.info(f"Scheduled collection completed for term: {term}")
            time.sleep(60)  # Wait between collections to avoid rate limiting
    except Exception as e:
        logger.error(f"Error in scheduled data collection: {str(e)}")

# Start scheduled jobs
def start_scheduler():
    """Start the scheduler for periodic data collection"""
    schedule.every().day.at("02:00").do(scheduled_data_collection)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    # Run in a separate thread
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    logger.info("Scheduler started for automated data collection")

# Function to create a choropleth map visualization
def create_choropleth_map(data, term):
    """Create a choropleth map for visualizing trends by state"""
    if data.empty:
        return None
        
    fig = px.choropleth(
        data,
        locations='state',
        locationmode='USA-states',
        color='interest',
        scope='usa',
        color_continuous_scale='Viridis',
        title=f'Search Interest for "{term}" by State',
        labels={'interest': 'Search Interest'},
        hover_data=['state', 'interest']
    )
    
    fig.update_layout(
        geo=dict(
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        ),
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    return fig

# Function to create correlation analysis visualization
def create_correlation_analysis(trends_data, demographic_data, term):
    """Create correlation analysis between search interest and demographic factors"""
    if trends_data.empty or demographic_data.empty:
        return None
        
    # Merge trends data with demographic data
    merged = pd.merge(
        trends_data,
        demographic_data,
        left_on='state',
        right_on='state',
        how='inner'
    )
    
    if merged.empty:
        return None
        
    # Calculate correlation coefficients
    corr_population = merged['interest'].corr(merged['population'])
    corr_age = merged['interest'].corr(merged['median_age'])
    corr_income = merged['interest'].corr(merged['income_per_capita'])
    corr_healthcare = merged['interest'].corr(merged['healthcare_coverage_pct'])
    
    # Create bar chart for correlation coefficients
    corr_data = pd.DataFrame({
        'Factor': ['Population', 'Median Age', 'Income Per Capita', 'Healthcare Coverage'],
        'Correlation': [corr_population, corr_age, corr_income, corr_healthcare]
    })
    
    fig = px.bar(
        corr_data,
        x='Factor',
        y='Correlation',
        title=f'Correlation of "{term}" Search Interest with Demographic Factors',
        labels={'Correlation': 'Correlation Coefficient'},
        color='Correlation',
        color_continuous_scale=px.colors.diverging.RdBu,
        range_color=[-1, 1]
    )
    
    return fig

# Function to create trend over time visualization
def create_trend_over_time(term, timeframe='today 12-m'):
    """Create trend over time visualization for a search term"""
    try:
        pytrends = get_pytrends_client()
        pytrends.build_payload([term], timeframe=timeframe)
        data = pytrends.interest_over_time()
        
        if data.empty:
            return None
            
        fig = px.line(
            data,
            x=data.index,
            y=term,
            title=f'Search Interest for "{term}" Over Time',
            labels={term: 'Search Interest', 'date': 'Date'}
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating trend over time visualization: {str(e)}")
        return None

# Function to create related queries visualization
def create_related_queries(term):
    """Create visualization for related queries"""
    try:
        pytrends = get_pytrends_client()
        pytrends.build_payload([term])
        data = pytrends.related_queries()
        
        if not data or term not in data or 'top' not in data[term] or data[term]['top'] is None:
            return None
            
        top_queries = data[term]['top'].head(10)
        
        fig = px.bar(
            top_queries,
            x='value',
            y='query',
            orientation='h',
            title=f'Top Related Queries for "{term}"',
            labels={'value': 'Search Interest', 'query': 'Query'}
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating related queries visualization: {str(e)}")
        return None

# Mock demographic data for testing
def load_mock_demographic_data():
    """Load mock demographic data if real data is not available"""
    mock_data = {
        'state': [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
            'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia'
        ],
        'population': [
            4903185, 731545, 7278717, 3017804, 39512223, 
            5758736, 3565287, 973764, 21477737, 10617423
        ],
        'median_age': [
            38.6, 34.6, 37.9, 38.0, 36.7, 
            36.9, 41.0, 40.7, 42.0, 36.9
        ],
        'income_per_capita': [
            27928, 35065, 29265, 26577, 35021, 
            38226, 43781, 34199, 30197, 29523
        ],
        'healthcare_coverage_pct': [
            89.6, 86.5, 88.3, 91.2, 92.7, 
            93.5, 94.8, 93.2, 86.1, 85.2
        ],
        'last_updated': [datetime.now()] * 10
    }
    return pd.DataFrame(mock_data)

# Save mock demographic data to database for testing
def save_mock_demographic_data():
    """Save mock demographic data to database"""
    try:
        engine = init_db()
        if engine is None:
            return False
            
        df = load_mock_demographic_data()
        df.to_sql('demographic_data', engine, if_exists='replace', index=False)
        logger.info("Mock demographic data saved to database")
        return True
    except Exception as e:
        logger.error(f"Error saving mock demographic data: {str(e)}")
        return False

# Main application UI
def main():
    # Sidebar
    st.sidebar.title("Healthcare Search Trends")
    st.sidebar.image("https://www.cdc.gov/healthliteracy/images/HL-Tools-for-Organizations-Card.png", width=250)
    
    section = st.sidebar.radio(
        "Navigation",
        ["Search Trends Analysis", "Correlation Analysis", "Trend Over Time", "Related Queries", "Data Management"]
    )
    
    # Initialize database
    engine = init_db()
    if engine is None:
        st.warning("Database connection failed. Some features may not work properly.")
    
    # Check for demographic data and populate with mock data if needed
    demo_data = load_demographic_data()
    if demo_data.empty:
        st.sidebar.warning("No demographic data found. Using mock data for analysis.")
        save_mock_demographic_data()
    
    if section == "Search Trends Analysis":
        st.title("Healthcare Search Trends Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Form for searching trends
            with st.form("search_form"):
                search_term = st.text_input("Enter a healthcare-related search term", "diabetes")
                region = st.selectbox("Select region", ["US", "Global"])
                timeframe = st.selectbox(
                    "Select timeframe",
                    ["today 1-m", "today 3-m", "today 12-m", "today 5-y"]
                )
                submit_button = st.form_submit_button("Search")
                
                if submit_button:
                    with st.spinner("Fetching data from Google Trends..."):
                        data = get_google_trends_data_with_retry([search_term], geo=region, timeframe=timeframe)
                        
                        if not data.empty:
                            st.success(f"Data retrieved for '{search_term}'")
                            store_trends_data(search_term, region, data)
                            
                            # Process data for visualization
                            data = data.reset_index()
                            data.columns = ['state', 'interest']
                            
                            # Create choropleth map
                            fig = create_choropleth_map(data, search_term)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Unable to create map visualization")
                        else:
                            st.error("No data found for this search term and region")
        
        with col2:
            st.subheader("Popular Health Topics")
            default_terms = [
                "COVID-19", "diabetes", "heart disease", "cancer", 
                "mental health", "anxiety", "depression", "asthma", 
                "obesity", "nutrition"
            ]
            
            for term in default_terms:
                if st.button(term, key=f"btn_{term}"):
                    st.session_state.search_term = term
                    st.rerun()
    
    elif section == "Correlation Analysis":
        st.title("Correlation Analysis")
        
        # Load tracked terms
        tracked_terms = get_tracked_terms()
        if tracked_terms.empty:
            st.warning("No tracked terms found. Please search for terms in the Search Trends Analysis section first.")
        else:
            term = st.selectbox("Select a healthcare term", tracked_terms['term'].tolist())
            
            if term:
                with st.spinner("Loading data..."):
                    # Load trends data
                    trends_data = load_trends_data(term=term, region='US')
                    
                    # Load demographic data
                    demo_data = load_demographic_data()
                    
                    if not trends_data.empty and not demo_data.empty:
                        # Group by state and get the most recent data point for each state
                        trends_data = trends_data.sort_values('timestamp', ascending=False)
                        trends_data = trends_data.drop_duplicates(subset=['state'])
                        
                        # Create correlation analysis
                        fig = create_correlation_analysis(trends_data, demo_data, term)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create scatter plots for each demographic factor
                            st.subheader("Detailed Correlation Plots")
                            
                            merged = pd.merge(
                                trends_data,
                                demo_data,
                                left_on='state',
                                right_on='state',
                                how='inner'
                            )
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig1 = px.scatter(
                                    merged,
                                    x='healthcare_coverage_pct',
                                    y='interest',
                                    hover_data=['state'],
                                    title=f"Healthcare Coverage vs. '{term}' Interest",
                                    labels={
                                        'healthcare_coverage_pct': 'Healthcare Coverage (%)',
                                        'interest': 'Search Interest'
                                    }
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                                
                                fig2 = px.scatter(
                                    merged,
                                    x='median_age',
                                    y='interest',
                                    hover_data=['state'],
                                    title=f"Median Age vs. '{term}' Interest",
                                    labels={
                                        'median_age': 'Median Age',
                                        'interest': 'Search Interest'
                                    }
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            with col2:
                                fig3 = px.scatter(
                                    merged,
                                    x='income_per_capita',
                                    y='interest',
                                    hover_data=['state'],
                                    title=f"Income per Capita vs. '{term}' Interest",
                                    labels={
                                        'income_per_capita': 'Income per Capita ($)',
                                        'interest': 'Search Interest'
                                    }
                                )
                                st.plotly_chart(fig3, use_container_width=True)
                                
                                fig4 = px.scatter(
                                    merged,
                                    x='population',
                                    y='interest',
                                    hover_data=['state'],
                                    title=f"Population vs. '{term}' Interest",
                                    labels={
                                        'population': 'Population',
                                        'interest': 'Search Interest'
                                    }
                                )
                                st.plotly_chart(fig4, use_container_width=True)
                        else:
                            st.warning("Unable to create correlation analysis")
                    else:
                        st.warning("Insufficient data for correlation analysis")
    
    elif section == "Trend Over Time":
        st.title("Trend Over Time Analysis")
        
        # Load tracked terms
        tracked_terms = get_tracked_terms()
        if tracked_terms.empty:
            st.warning("No tracked terms found. Please search for terms in the Search Trends Analysis section first.")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                term = st.selectbox("Select a healthcare term", tracked_terms['term'].tolist())
                timeframe = st.selectbox(
                    "Select timeframe",
                    ["today 1-m", "today 3-m", "today 12-m", "today 5-y"]
                )
                
                if term:
                    with st.spinner("Loading trend data..."):
                        fig = create_trend_over_time(term, timeframe)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Unable to create trend visualization")
            
            with col2:
                st.subheader("Trend Insights")
                st.info("""
                This chart shows how search interest for the selected healthcare term has changed over time.
                
                A value of 100 is the peak popularity for the term. A value of 50 means that the term is half as popular.
                
                Peaks may indicate:
                - Disease outbreaks
                - Seasonal health issues
                - Major health news
                - Policy changes
                """)
    
    elif section == "Related Queries":
        st.title("Related Queries Analysis")
        
        # Load tracked terms
        tracked_terms = get_tracked_terms()
        if tracked_terms.empty:
            st.warning("No tracked terms found. Please search for terms in the Search Trends Analysis section first.")
        else:
            term = st.selectbox("Select a healthcare term", tracked_terms['term'].tolist())
            
            if term:
                with st.spinner("Loading related queries..."):
                    fig = create_related_queries(term)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Try to get rising queries too
                        try:
                            pytrends = get_pytrends_client()
                            pytrends.build_payload([term])
                            data = pytrends.related_queries()
                            
                            if term in data and 'rising' in data[term] and data[term]['rising'] is not None:
                                rising = data[term]['rising'].head(10)
                                
                                st.subheader("Rising Related Queries")
                                fig2 = px.bar(
                                    rising,
                                    x='value',
                                    y='query',
                                    orientation='h',
                                    title=f"Rising Related Queries for '{term}'",
                                    labels={'value': 'Growth', 'query': 'Query'}
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                        except Exception as e:
                            logger.error(f"Error getting rising queries: {str(e)}")
                    else:
                        st.warning("No related queries found for this term")
    
    elif section == "Data Management":
        st.title("Data Management")
        
        tab1, tab2, tab3 = st.tabs(["Tracked Terms", "Export Data", "Database Maintenance"])
        
        with tab1:
            st.subheader("Tracked Terms")
            tracked_terms = get_tracked_terms()
            
            if not tracked_terms.empty:
                st.dataframe(tracked_terms, use_container_width=True)
                
                # Delete selected terms
                if st.button("Delete Selected Terms"):
                    st.warning("Feature not implemented yet")
            else:
                st.warning("No tracked terms found")
        
        with tab2:
            st.subheader("Export Data")
            
            # Load tracked terms for export
            tracked_terms = get_tracked_terms()
            if not tracked_terms.empty:
                term = st.selectbox("Select term to export", tracked_terms['term'].tolist(), key="export_term")
                
                if term:
                    data = load_trends_data(term=term)
                    if not data.empty:
                        st.dataframe(data.head(), use_container_width=True)
                        
                        # Export options
                        export_format = st.radio("Export format", ["CSV", "Excel", "JSON"])
                        
                        if st.button("Export Data"):
                            if export_format == "CSV":
                                csv = data.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f"{term}_trends_data.csv",
                                    mime="text/csv"
                                )
                            elif export_format == "Excel":
                                st.warning("Excel export requires additional libraries. Please use CSV instead.")
                            elif export_format == "JSON":
                                json_data = data.to_json(orient="records")
                                st.download_button(
                                    label="Download JSON",
                                    data=json_data,
                                    file_name=f"{term}_trends_data.json",
                                    mime="application/json"
                                )
                    else:
                        st.warning("No data found for this term")
            else:
                st.warning("No tracked terms found")
        
        with tab3:
            st.subheader("Database Maintenance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Check Database Health"):
                    try:
                        engine = init_db()
                        if engine is not None:
                            with engine.connect() as conn:
                                result = conn.execute(text("PRAGMA integrity_check"))
                                integrity = result.scalar()
                                
                                if integrity == "ok":
                                    st.success("Database integrity check passed")
                                else:
                                    st.error(f"Database integrity check failed: {integrity}")
                        else:
                            st.error("Database connection failed")
                    except Exception as e:
                        st.error(f"Error checking database health: {str(e)}")
            
            with col2:
                if st.button("Refresh Demographic Data"):
                    if save_mock_demographic_data():
                        st.success("Demographic data refreshed")
                    else:
                        st.error("Failed to refresh demographic data")
            
            # Database statistics
            try:
                engine = init_db()
                if engine is not None:
                    st.subheader("Database Statistics")
                    
                    with engine.connect() as conn:
                        # Count trends data
                        result = conn.execute(text("SELECT COUNT(*) FROM trends_data"))
                        trends_count = result.scalar()
                        
                        # Count tracked terms
                        result = conn.execute(text("SELECT COUNT(*) FROM tracked_terms"))
                        terms_count = result.scalar()
                        
                        # Count demographic data
                        result = conn.execute(text("SELECT COUNT(*) FROM demographic_data"))
                        demo_count = result.scalar()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Trends Data Entries", trends_count)
                    col2.metric("Tracked Terms", terms_count)
                    col3.metric("Demographic Entries", demo_count)
            except Exception as e:
                st.error(f"Error getting database statistics: {str(e)}")

# Start the application
if __name__ == "__main__":
    try:
        # Initialize the database
        init_db()
        
        # Start the scheduler
        start_scheduler()
        
        # Run the main application
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
