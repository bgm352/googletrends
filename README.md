# Healthcare Search Trends Analyzer

A Streamlit application that analyzes healthcare-related search trends using Google Trends data.

## Features

- Search interest visualization across US states
- Correlation analysis with demographic factors
- Trend over time analysis
- Related queries analysis
- Data management and export capabilities

## Requirements

This application requires:

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- Requests
- PyTrends
- SQLAlchemy
- Schedule

## Installation

1. Clone this repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application with:

```bash
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Fork this repository
2. Connect your fork to Streamlit Cloud
3. Deploy directly from GitHub

## Troubleshooting

If you encounter import errors:
- Make sure all dependencies are listed in requirements.txt with exact versions
- Check the Streamlit Cloud logs for specific error messages
- Verify that all necessary packages are installed in your environment

## Data Storage

The application uses SQLite for data storage. In Streamlit Cloud, the database will be ephemeral (reset on each deployment). For persistent storage, consider using:

- External database services
- Cloud storage solutions
- Streamlit's built-in caching mechanisms

## License

MIT
