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


