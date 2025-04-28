# Healthcare Search Trends Analyzer

## Overview
This Streamlit application helps healthcare organizations analyze Google Trends data to create comprehensive heatmaps showing geographic distribution of healthcare-related searches. The tool enables health systems to visualize where specific conditions, treatments, or services are being researched most actively, helping marketers identify regions with high demand but limited service availability.

## Features
- **Search Term Analysis**: Analyze predefined healthcare terms or input custom search queries
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

## Usage

1. Select either predefined healthcare terms or enter a custom search query
2. Choose the geographic region and time frame for analysis
3. Select the visualization type (relative interest or population-weighted)
4. Click "Analyze Search Trends" to generate insights
5. Navigate through the tabs to explore different visualizations and insights
6. Download data for further analysis or reporting

## Data Sources

- **Google Trends**: Search interest data across geographic regions
- **Mock Demographic Data**: For demonstration purposes, the app uses mock demographic data. In a production environment, this would be replaced with real demographic data from sources like the US Census Bureau.

## Limitations

- Google Trends data is relative, not absolute search volume
- The PyTrends API has rate limitations that may affect frequent usage
- Demographic data in this demo version is synthetic

## Future Enhancements

- Integration with real healthcare coverage data
- Competitor analysis features
- Time-series analysis of changing search patterns
- Integration with Google Analytics for healthcare websites
- Marketing ROI calculator based on search trends
