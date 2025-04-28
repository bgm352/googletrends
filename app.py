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
                    'category': 'custom',  # Categorize automatically in a more advanced version
                    'added_date': timestamp,
                    'last_updated': timestamp
                })
            
            logger.info(f"Stored trends data for {term} in {region}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error storing trends data: {str(e)}")
        return False

# Get trends data from database or API
@st.cache_data(ttl=3600)
def get_trends_data(term, region, timeframe='today 12-m', use_cache=True):
    """Get trends data from database if available or from API"""
    try:
        engine = init_db()
        if engine is None:
            return get_google_trends_data_with_retry([term], region, timeframe)
            
        # Check if recent data exists in database
        if use_cache:
            cache_threshold = datetime.now() - timedelta(days=1)  # Use cache if data is less than 1 day old
            query = text('''
                SELECT state, interest, timestamp
                FROM trends_data
                WHERE term = :term AND region = :region AND timestamp > :threshold
                ORDER BY timestamp DESC
            ''')
            
            with engine.connect() as conn:
                result = conn.execute(query, {
                    'term': term,
                    'region': region,
                    'threshold': cache_threshold
                }).fetchall()
            
            if result:
                # Convert to DataFrame
                logger.info(f"Using cached data for {term} in {region}")
                df = pd.DataFrame(result, columns=['state', 'interest', 'timestamp'])
                trends_data = pd.DataFrame(index=df['state'])
                trends_data[term] = df['interest'].values
                return trends_data
        
        # If no recent data or cache not requested, get from API
        trends_data = get_google_trends_data_with_retry([term], region, timeframe)
        
        # Store the new data
        if not trends_data.empty:
            store_trends_data(term, region, trends_data)
            
        return trends_data
    except Exception as e:
        logger.error(f"Error in get_trends_data: {str(e)}")
        return get_google_trends_data_with_retry([term], region, timeframe)

# Function to load demographic data (real API or fallback)
def load_demographic_data():
    try:
        engine = init_db()
        if engine is None:
            return load_mock_demographic_data()
            
        # Check if we have demographic data in the database
        with engine.connect() as conn:
            result = conn.execute(text('''
                SELECT * FROM demographic_data
                ORDER BY last_updated DESC
                LIMIT 1
            ''')).fetchone()
            
        if result:
            # Data exists in database, check if it's recent
            cache_threshold = datetime.now() - timedelta(days=30)  # Demographic data refreshed monthly
            if result.last_updated > cache_threshold:
                # Use cached demographic data
                query = text('SELECT * FROM demographic_data')
                with engine.connect() as conn:
                    demo_data = pd.read_sql(query, conn)
                logger.info("Using cached demographic data")
                return demo_data
        
        # Try to get real data from Census API
        try:
            # Uncomment and use when you have a Census API key
            """
            from census import Census
            from us import states
            
            c = Census("YOUR_CENSUS_API_KEY")
            demographics = c.acs5.state(
                ('NAME', 'B01003_001E', 'B06002_001E', 'B19301_001E'),  # Population, Median Age, Per Capita Income
                Census.ALL
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(demographics)
            df = df.rename(columns={
                'NAME': 'state',
                'B01003_001E': 'population',
                'B06002_001E': 'median_age',
                'B19301_001E': 'income_per_capita'
            })
            
            # Add healthcare coverage data from HHS or other source
            healthcare_coverage = {
                'Massachusetts': 97.5, 'Hawaii': 96.2, 'Vermont': 95.8,
                # Add other states as needed
            }
            
            df['healthcare_coverage_pct'] = df['state'].apply(
                lambda x: healthcare_coverage.get(x, 87.0)  # US average ~87%
            )
            
            df['last_updated'] = datetime.now()
            df.to_sql('demographic_data', engine, if_exists='replace', index=False)
            
            logger.info("Retrieved and stored new demographic data from Census API")
            return df
            """
            # For now, use the CSV file if available
            if os.path.exists('demographic_data.csv'):
                df = pd.read_csv('demographic_data.csv')
                df['last_updated'] = datetime.now()
                df.to_sql('demographic_data', engine, if_exists='replace', index=False)
                logger.info("Loaded demographic data from CSV file")
                return df
            else:
                return load_mock_demographic_data()
                
        except Exception as e:
            logger.warning(f"Failed to get real demographic data: {str(e)}. Using mock data.")
            return load_mock_demographic_data()
    except Exception as e:
        logger.error(f"Error in load_demographic_data: {str(e)}")
        return load_mock_demographic_data()

# Mock demographic data generator
def load_mock_demographic_data():
    """Generate mock demographic data for demonstration purposes"""
    states = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
        "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
        "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
        "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
        "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
        "Wisconsin", "Wyoming"
    ]
    
    np.random.seed(42)  # For reproducible demo data
    
    demographics = pd.DataFrame({
        'state': states,
        'population': np.random.randint(500000, 40000000, size=len(states)),
        'median_age': np.random.randint(30, 50, size=len(states)),
        'income_per_capita': np.random.randint(25000, 75000, size=len(states)),
        'healthcare_coverage_pct': np.random.randint(70, 98, size=len(states)),
        'last_updated': datetime.now()
    })
    
    # Store in database
    try:
        engine = init_db()
        if engine:
            demographics.to_sql('demographic_data', engine, if_exists='replace', index=False)
    except Exception as e:
        logger.error(f"Failed to store mock demographic data: {str(e)}")
        
    logger.info("Generated mock demographic data")
    return demographics

# Function to combine Google Trends data with demographics
def combine_with_demographics(trends_data, demographics):
    """Combine trends data with demographic information"""
    if trends_data.empty:
        return pd.DataFrame()
    
    # Reset index to make 'state' a column
    trends_data = trends_data.reset_index()
    # The column from pytrends is 'geoName' for states
    if 'geoName' in trends_data.columns:
        trends_data = trends_data.rename(columns={'geoName': 'state'})
    
    # Merge with demographic data
    combined = pd.merge(trends_data, demographics, on='state', how='left')
    return combined

# Function to create a heatmap using Plotly
def create_heatmap(data, search_term, metric='relative_interest'):
    """Create a choropleth map visualization of search trends"""
    if data.empty:
        return None
    
    # Create a heatmap using Plotly Express
    if metric == 'relative_interest':
        fig = px.choropleth(
            data,
            locations='state',
            locationmode='USA-states',
            color=search_term,
            scope='usa',
            color_continuous_scale='Viridis',
            title=f'Geographic Distribution of "{search_term}" Searches',
            labels={search_term: 'Relative Interest (0-100)'}
        )
    elif metric == 'weighted_interest':
        # Calculate weighted interest (interest × population)
        data['weighted_interest'] = data[search_term] * data['population'] / 1000000
        fig = px.choropleth(
            data,
            locations='state',
            locationmode='USA-states',
            color='weighted_interest',
            scope='usa',
            color_continuous_scale='Viridis',
            title=f'Population-Weighted Interest in "{search_term}" Searches',
            labels={'weighted_interest': f'Weighted Interest ({search_term} × Pop in millions)'}
        )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title="Search Interest",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300
        )
    )
    
    return fig

# Function to get related queries with caching
@st.cache_data(ttl=3600)
def get_related_queries_with_retry(keywords, geo, timeframe='today 12-m', max_retries=3):
    """Get related queries data with retry logic"""
    retries = 0
    while retries < max_retries:
        try:
            pytrends = get_pytrends_client()
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
            return pytrends.related_queries()
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                retries += 1
                wait_time = 60 * retries
                logger.warning(f"Rate limited when getting related queries. Retry {retries}/{max_retries} in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error fetching related queries: {str(e)}")
                return {}
    logger.error("Failed to retrieve related queries after multiple attempts")
    return {}


