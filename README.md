# Healthcare Search Trends Analyzer

## Overview
This Streamlit application helps healthcare organizations analyze Google Trends data to create comprehensive heatmaps showing geographic distribution of healthcare-related searches. The tool enables health systems to visualize where specific conditions, treatments, or services are being researched most actively, helping marketers identify regions with high demand but limited service availability.

## Features
- **Real-time Google Trends Data**: Access actual Google Trends data through the PyTrends API
- **Local Database Storage**: Cache search trends data for improved performance and historical analysis
- **Automated Data Updates**: Background job to refresh data for tracked terms
- **Geographic Heatmaps**: Visualize search interest across different US states
- **Demographic Integration**: Combine search data with population and healthcare coverage information
- **Market Insights**: Identify high-opportunity markets based on search interest and healthcare coverage
- **Related Search Analysis**: View related queries to understand consumer interest patterns
- **Data Export**: Download raw data for further analysis

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/healthcare-search-trends.git
cd healthcare-search-trends
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. (Optional) Schedule background updates:
```bash
# Set up a cron job or scheduled task to run:
python data_updater.py
```

## Usage

1. Select either predefined healthcare terms or enter a custom search query
2. Choose the geographic region and time frame for analysis
3. Select the visualization type (relative interest or population-weighted)
4. Choose whether to use cached data or fetch new data
5. Click "Analyze Search Trends" to generate insights
6. Navigate through the tabs to explore different visualizations and insights
7. Download data for further analysis or reporting

## Data Sources

- **Google Trends**: Real search interest data across geographic regions via PyTrends
- **Database Storage**: SQLite database for caching and historical analysis
- **Demographic Data**: The application can work with:
  - Census API data (requires API key)
  - CSV import of demographic data
  - Mock demographic data for demonstration purposes

## Rate Limiting and Performance Considerations

Google Trends has API rate limits that this application respects through:
- Caching of results
- Rate limiting with exponential backoff
- Scheduled updates to spread requests over time

## Customization

### Adding Real Census Data

To use real Census data:
1. Get a Census API key from [census.gov](https://api.census.gov/data/key_signup.html)
2. Uncomment and update the Census API code in the `load_demographic_data()` function
3. Add your API key where indicated

### Scheduling Background Updates

For production use, you can:
1. Use the built-in scheduler that runs when the app is active
2. Set up a separate cron job or scheduled task to run `data_updater.py` on a regular schedule

## Limitations

- Google Trends data is relative, not absolute search volume
- The PyTrends API has rate limitations that may affect frequent usage
- Demographic data requires a Census API key for complete accuracy

## Files
- `app.py`: Main Streamlit application
- `data_updater.py`: Standalone script for background updates
- `requirements.txt`: Python package dependencies
- `README.md`: This documentation file
