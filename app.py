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
            # Log successful retrieval
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
                df = pd.DataFrame(result)
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
            # This is placeholder - in production, get real healthcare coverage data
            # For now use realistic values based on state rankings
            healthcare_coverage = {
                # Values from Kaiser Family Foundation data (example)
                'Massachusetts': 97.5, 'Hawaii': 96.2, 'Vermont': 95.8, 
                # More states would be added here
            }
            
            # Set default coverage for states not in our data
            df['healthcare_coverage_pct'] = df['state'].apply(
                lambda x: healthcare_coverage.get(x, 87.0)  # US average ~87%
            )
            
            # Store in database for future use
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
                st.error(f"Error fetching related queries: {str(e)}")
                return {}
    
    logger.error("Failed to retrieve related queries after multiple attempts")
    st.error("Failed to retrieve related queries. Try again later.")
    return {}

# Function to get related topics with caching
@st.cache_data(ttl=3600)
def get_related_topics_with_retry(keywords, geo, timeframe='today 12-m', max_retries=3):
    """Get related topics data with retry logic"""
    retries = 0
    while retries < max_retries:
        try:
            pytrends = get_pytrends_client()
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
            return pytrends.related_topics()
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                retries += 1
                wait_time = 60 * retries
                logger.warning(f"Rate limited when getting related topics. Retry {retries}/{max_retries} in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error fetching related topics: {str(e)}")
                st.error(f"Error fetching related topics: {str(e)}")
                return {}
    
    logger.error("Failed to retrieve related topics after multiple attempts")
    st.error("Failed to retrieve related topics. Try again later.")
    return {}

# Load predefined healthcare categories and terms
def load_healthcare_categories():
    """Load healthcare categories and terms"""
    # First check if we have them in the database
    try:
        engine = init_db()
        if engine:
            with engine.connect() as conn:
                query = text('''
                SELECT term, category FROM tracked_terms
                WHERE category != 'custom'
                ORDER BY category, term
                ''')
                result = conn.execute(query).fetchall()
                
            if result:
                # Convert to dictionary of categories
                categories = {}
                for row in result:
                    category = row.category
                    term = row.term
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(term)
                
                # If we have at least some predefined categories
                if len(categories) >= 4:
                    logger.info("Loaded healthcare categories from database")
                    return categories
    except Exception as e:
        logger.error(f"Error loading categories from database: {str(e)}")
    
    # Default categories if not in database
    categories = {
        "Conditions": [
            "Diabetes", "Hypertension", "Cancer", "Heart disease", "Asthma", 
            "Arthritis", "Alzheimer's", "Depression", "Anxiety", "COVID-19"
        ],
        "Treatments": [
            "Surgery", "Physical therapy", "Chemotherapy", "Radiation therapy", 
            "Immunotherapy", "Telemedicine", "Telehealth", "Remote patient monitoring"
        ],
        "Specialties": [
            "Cardiology", "Oncology", "Neurology", "Orthopedics", "Pediatrics", 
            "Obstetrics", "Gynecology", "Dermatology", "Psychiatry"
        ],
        "Healthcare Services": [
            "Urgent care", "Emergency room", "Primary care", "Hospital", 
            "Clinic", "Rehabilitation", "Home health care", "Hospice"
        ]
    }
    
    # Store in database for future use
    try:
        engine = init_db()
        if engine:
            timestamp = datetime.now()
            for category, terms in categories.items():
                for term in terms:
                    with engine.connect() as conn:
                        conn.execute(text('''
                        INSERT INTO tracked_terms (term, category, added_date, last_updated)
                        VALUES (:term, :category, :added_date, :last_updated)
                        ON CONFLICT(term) DO UPDATE SET
                        category = excluded.category,
                        last_updated = excluded.last_updated
                        '''), {
                            'term': term,
                            'category': category,
                            'added_date': timestamp,
                            'last_updated': timestamp
                        })
            logger.info("Stored default healthcare categories in database")
    except Exception as e:
        logger.error(f"Failed to store healthcare categories: {str(e)}")
    
    return categories

# Background job to update trends data
def update_trends_database():
    """Background job to update trends data for tracked terms"""
    try:
        engine = init_db()
        if engine is None:
            logger.error("Database not available for background update")
            return
            
        # Get list of terms to track from database
        with engine.connect() as conn:
            query = text('''
            SELECT term FROM tracked_terms
            ORDER BY last_updated ASC
            LIMIT 10  -- Update oldest 10 terms to avoid rate limiting
            ''')
            terms_to_track = [row[0] for row in conn.execute(query).fetchall()]
        
        if not terms_to_track:
            logger.info("No terms to update in background job")
            return
            
        regions = ["US"]  # Could expand to more regions
        
        for term in terms_to_track:
            for region in regions:
                # Get fresh data with a longer timeframe
                logger.info(f"Background update for term: {term} in {region}")
                data = get_google_trends_data_with_retry([term], geo=region)
                # Store in database
                if not data.empty:
                    store_trends_data(term, region, data)
                # Sleep to avoid rate limiting
                time.sleep(10)
        
        logger.info(f"Updated trends data for {len(terms_to_track)} terms at {datetime.now()}")
    except Exception as e:
        logger.error(f"Error in background update job: {str(e)}")

# Background threads manager
def background_updater():
    """Manage scheduled background jobs"""
    while True:
        schedule.run_pending()
        time.sleep(60)

# Initialize database and start background thread when app loads
engine = init_db()
if engine and not st.session_state.get('background_started', False):
    # Schedule jobs
    schedule.every().day.at("02:00").do(update_trends_database)
    
    # Start background thread
    threading.Thread(target=background_updater, daemon=True).start()
    st.session_state['background_started'] = True
    logger.info("Started background updater thread")

# Main application
def main():
    st.title("🏥 Healthcare Search Trends Analyzer")
    
    st.markdown("""
    This application helps healthcare marketers visualize Google Trends data to identify geographic 
    hotspots for healthcare-related searches. The insights can guide marketing strategies, resource 
    allocation, and potential new facility locations.
    """)
    
    # Check database status
    db_status = "Connected" if engine else "Disconnected"
    st.sidebar.text(f"Database: {db_status}")
    
    # Sidebar for input controls
    st.sidebar.header("Search Parameters")
    
    # Tab selection
    tab_options = ["Predefined Terms", "Custom Search"]
    selected_tab = st.sidebar.radio("Choose Search Method:", tab_options)
    
    if selected_tab == "Predefined Terms":
        # Load healthcare categories
        categories = load_healthcare_categories()
        
        # Category selection
        selected_category = st.sidebar.selectbox(
            "Select Healthcare Category:", 
            list(categories.keys())
        )
        
        # Term selection based on category
        selected_term = st.sidebar.selectbox(
            "Select Specific Term:", 
            categories[selected_category]
        )
        
        search_term = selected_term
    else:
        # Get previously searched custom terms
        custom_terms = []
        try:
            if engine:
                with engine.connect() as conn:
                    query = text('''
                    SELECT DISTINCT term FROM trends_data 
                    WHERE term NOT IN (SELECT term FROM tracked_terms WHERE category != 'custom')
                    ORDER BY term
                    ''')
                    result = conn.execute(query).fetchall()
                    if result:
                        custom_terms = [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error fetching custom terms: {str(e)}")
        
        # Custom search term input with suggestions
        if custom_terms:
            search_term = st.sidebar.selectbox(
                "Select or Enter Healthcare Search Term:", 
                ["telehealth"] + custom_terms,
                index=0
            )
        else:
            search_term = st.sidebar.text_input("Enter Healthcare Search Term:", "telehealth")
    
    # Region selection
    region = st.sidebar.selectbox(
        "Select Region:",
        ["US", "CA", "GB", "AU"]
    )
    
    # Time frame selection
    time_options = {
        "Past 12 months": "today 12-m",
        "Past 3 months": "today 3-m", 
        "Past 30 days": "today 1-m",
        "Past 7 days": "now 7-d"
    }
    
    timeframe = st.sidebar.selectbox(
        "Select Time Frame:",
        list(time_options.keys())
    )
    
    # Data source option
    use_cache = st.sidebar.checkbox("Use cached data when available", value=True, 
                                   help="Improves performance but may not show the very latest trends")
    
    # Visualization type
    viz_type = st.sidebar.radio(
        "Select Visualization Type:",
        ["Relative Interest", "Population-Weighted Interest"]
    )
    
    viz_metric = "relative_interest" if viz_type == "Relative Interest" else "weighted_interest"
    
    # Button to trigger analysis
    analyze_button = st.sidebar.button("Analyze Search Trends")
    
    # Main content area
    if analyze_button:
        with st.spinner(f"Analyzing search trends for '{search_term}'..."):
            # Get trends data (from cache or API)
            trends_data = get_trends_data(search_term, region, time_options[timeframe], use_cache)
            
            if not trends_data.empty:
                # Get demographic data
                demographics = load_demographic_data()
                
                # Combine trends with demographics
                combined_data = combine_with_demographics(trends_data, demographics)
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Geographic Heatmap", "Data Table", "Related Searches", "Market Insights"])
                
                with tab1:
                    st.subheader(f"Geographic Distribution of '{search_term}' Searches")
                    
                    # Create and display the heatmap
                    fig = create_heatmap(combined_data, search_term, metric=viz_metric)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Unable to create heatmap with the provided data.")
                
                with tab2:
                    st.subheader("Raw Data Table")
                    
                    # Add timestamp info
                    if 'timestamp' in combined_data.columns:
                        data_date = combined_data['timestamp'].iloc[0]
                        st.info(f"Data retrieved: {data_date}")
                    
                    st.dataframe(combined_data)
                    
                    # Download button for CSV
                    csv = combined_data.to_csv(index=False)
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name=f"{search_term}_trends_data.csv",
                        mime="text/csv",
                    )
                
                with tab3:
                    st.subheader("Related Searches")
                    
                    # Get related queries
                    related_queries = get_related_queries_with_retry([search_term], region, time_options[timeframe])
                    
                    if related_queries and search_term in related_queries:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("##### Top Related Queries")
                            if 'top' in related_queries[search_term] and not related_queries[search_term]['top'].empty:
                                st.dataframe(related_queries[search_term]['top'].head(10))
                            else:
                                st.write("No top related queries found.")
                        
                        with col2:
                            st.write("##### Rising Related Queries")
                            if 'rising' in related_queries[search_term] and not related_queries[search_term]['rising'].empty:
                                st.dataframe(related_queries[search_term]['rising'].head(10))
                            else:
                                st.write("No rising related queries found.")
                    else:
                        st.write("No related queries data available.")
                
                with tab4:
                    st.subheader("Market Insights")
                    
                    # Calculate high-opportunity markets (high interest, lower competition)
                    if not combined_data.empty:
                        # Calculate a simplified opportunity score
                        # High interest + lower healthcare coverage = potential opportunity
                        combined_data['opportunity_score'] = (
                            combined_data[search_term] * 
                            (100 - combined_data['healthcare_coverage_pct']) / 100
                        )
                        
                        # Display top opportunity states
                        st.write("##### Top Opportunity Markets")
                        top_opportunities = combined_data.sort_values('opportunity_score', ascending=False).head(10)
                        st.dataframe(top_opportunities[['state', search_term, 'healthcare_coverage_pct', 'opportunity_score']])
                        
                        # Create a bar chart of opportunity scores
                        fig = px.bar(
                            top_opportunities,
                            x='state',
                            y='opportunity_score',
                            color=search_term,
                            labels={'opportunity_score': 'Market Opportunity Score', 'state': 'State'},
                            title='Top 10 Market Opportunities by State'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations based on data
                        st.write("##### Strategic Recommendations")
                        st.write(f"""
                        Based on the analysis of search trends for "{search_term}", we recommend:
                        
                        1. **Focused Marketing:** Consider increasing marketing efforts in {top_opportunities['state'].iloc[0]} and {top_opportunities['state'].iloc[1]}, which show high interest with potential market gaps.
                        
                        2. **Service Expansion:** Evaluate service line expansion in regions with high search interest but lower healthcare coverage.
                        
                        3. **Content Strategy:** Create targeted content addressing related searches like "{related_queries[search_term]['top']['query'].iloc[0] if related_queries and search_term in related_queries and 'top' in related_queries[search_term] and not related_queries[search_term]['top'].empty else 'common questions'}" to capture additional relevant traffic.
                        """)
            else:
                st.error(f"No data available for the search term '{search_term}' in the selected region and time frame.")
    
    # Data management expander
    if engine:
        with st.sidebar.expander("Data Management"):
            if st.button("Run Manual Update"):
                with st.spinner("Updating database..."):
                    update_trends_database()
                st.success("Database update job triggered")
            
            # Show database stats
            try:
                with engine.connect() as conn:
                    term_count = conn.execute(text("SELECT COUNT(*) FROM tracked_terms")).scalar()
                    data_count = conn.execute(text("SELECT COUNT(*) FROM trends_data")).scalar()
                
                st.write(f"Tracked terms: {term_count}")
                st.write(f"Data records: {data_count}")
            except:
            st.write(f"Tracked terms: {term_count}")
                st.write(f"Data records: {data_count}")
            except Exception as e:
                st.write("Database statistics unavailable")
                logger.error(f"Error fetching database stats: {str(e)}")
    
    # Information section at the bottom
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this app:**
    
    This tool helps healthcare marketers identify geographic hotspots of interest for specific health conditions, treatments, or services. 
    By analyzing Google Trends data enhanced with demographic information, you can make data-driven decisions about where to focus marketing resources.
    """)

    # Display time of data refresh
    refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.caption(f"Last refresh: {refresh_time}")

# Error handling for the main app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
