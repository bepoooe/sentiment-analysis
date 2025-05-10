# Social Media Sentiment Analysis Tool

A sophisticated Flask-based web application for analyzing sentiment and engagement patterns in social media content, with a focus on Instagram data analysis.

## Features

### Data Analysis
- Comprehensive sentiment analysis of posts and comments using TextBlob
- Advanced engagement metrics calculation
- Machine learning-based engagement prediction
- Temporal analysis for optimal posting times
- Comment pattern recognition and categorization

### Technical Features
- Flask web interface with interactive visualizations
- Real-time data processing and analysis
- Caching system for performance optimization
- Robust error handling and logging
- Standardized metrics (0-100 scale)

### Visualization & Reporting
- Interactive Plotly dashboards
- Sentiment distribution charts
- Engagement heatmaps
- Day/hour performance analysis
- Natural language insights generation

## Requirements

### Core Dependencies
- Python 3.x
- Flask
- pandas
- numpy
- scikit-learn
- TextBlob
- Plotly
- joblib

### Data Format
- Instagram data in JSON format with the following structure:
  - Post metadata (id, timestamp, caption)
  - Engagement metrics (likes, comments)
  - Comment data (text, timestamp)

## Installation

1. Clone or download this repository

2. Install required packages:
```powershell
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
   - Place your Instagram JSON data files in the `instagram_data` folder
   - Files should be organized in timestamped directories: `account_name__YYYY-MM-DD_HH-MM-SS`

2. Start the Web Application:
```powershell
python app.py
```

3. Access the Interface:
   - Open your browser and navigate to `http://localhost:5000`
   - Select a data file to analyze
   - View interactive visualizations and insights

## Key Components

- `app.py`: Main Flask application and web interface
- `sentiment_analysis.py`: Core analysis functionality
- `models/engagement_model.joblib`: Pre-trained engagement prediction model
- `templates/`: HTML templates for web interface

## Output & Insights

The application provides:
- Interactive web dashboard with real-time analysis
- Sentiment and engagement visualizations
- Time-based performance analysis
- Actionable content recommendations
- Natural language insights about:
  - Content performance
  - Audience behavior
  - Optimal posting strategies
  - Engagement patterns
