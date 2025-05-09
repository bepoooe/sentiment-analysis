# Social Media Sentiment Analysis

A Python tool for analyzing sentiment and engagement patterns across Instagram and YouTube content.

## Features

- Multi-platform analysis (Instagram and YouTube)
- Sentiment analysis of posts and comments
- Engagement metrics calculation
- Content type performance analysis
- Best posting time analysis
- Cross-platform comparisons
- Detailed CSV exports for further analysis

## Requirements

- Python 3.x
- pandas
- textblob
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bepoooe/sentiment-analysis.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your Instagram JSON data in the `instagram_data` folder
2. Place your YouTube JSON data in the `youtube_data` folder
3. Run the analysis:
```bash
python sentiment_analysis.py
```

## Output

The script generates:
- Individual CSV files for each analyzed account
- Console output with detailed insights including:
  - Content style analysis
  - Audience behavior patterns
  - Engagement timing insights
  - Cross-platform performance comparison
