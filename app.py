from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import numpy as np
import json
import os
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from textblob import TextBlob
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants
DATA_FOLDER = 'instagram_data'
MODEL_PATH = 'models/engagement_model.joblib'
SENTIMENT_THRESHOLDS = {
    'very_positive': 80,
    'positive': 60,
    'neutral': 40,
    'negative': 20
}
COMMENT_EMOJIS = {
    'positive': ['❤️', '😍', '🔥', '👏', '💙', '😊', '👍', '💯', '🙌'],
    'negative': ['😢', '😭', '💔', '😞', '😪', '👎', '😡', '🤬']
}
COMMENT_KEYWORDS = {
    'enthusiastic': ['amazing', 'beautiful', 'awesome', 'great', 'love', 'excellent', 'perfect', 'fantastic'],
    'critical': ['fake', 'bad', 'terrible', 'hate', 'worst', 'poor', 'disappointing', 'awful']
}

def ensure_dir_exists(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

# Make sure model directory exists
ensure_dir_exists(os.path.dirname(MODEL_PATH))

def get_available_data_files(folder_path=DATA_FOLDER):
    """Get list of available data files with their paths."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_path, folder_path)
    available_files = []
    
    if os.path.exists(folder_path):
        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name)
            if os.path.isdir(dir_path):
                # Extract account name from directory name
                account_name = dir_name.split('_')[0]
                if account_name == 'blu':  # Handle special case for blu_es
                    account_name = 'blu_es'
                
                # Construct JSON filename
                json_file = f"{account_name}.json"
                if account_name == 'blu_es':
                    json_file = 'blu_es_.json'
                    
                json_path = os.path.join(dir_path, json_file)
                
                if os.path.exists(json_path):
                    try:
                        # Parse timestamp from directory name
                        # Format: account_name__YYYY-MM-DD_HH-MM-SS
                        timestamp_str = '_'.join(dir_name.split('__')[1:]).replace('-', ':')
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d:%H:%M:%S')
                    except (IndexError, ValueError):
                        timestamp = datetime.now()
                    
                    available_files.append({
                        'path': json_path,
                        'account_name': account_name,
                        'timestamp': timestamp,
                        'formatted_time': timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    })
    
    return sorted(available_files, key=lambda x: x['timestamp'], reverse=True)

def safe_parse_timestamp(timestamp_str, default=None):
    """Safely parse timestamp with multiple format attempts."""
    if not timestamp_str:
        return default or datetime.now()
        
    formats_to_try = [
        '%Y-%m-%dT%H:%M:%S.000Z',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S'
    ]
    
    for fmt in formats_to_try:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Unable to parse timestamp: {timestamp_str}, using default")
    return default or datetime.now()

def load_social_media_data(folder_path=DATA_FOLDER, platform='instagram', selected_file=None):
    """Enhanced data loading function that processes all JSON files in the folder structure."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_path, folder_path)
    logger.info(f"Looking for data in: {folder_path}")
    all_data = []
    
    if not os.path.exists(folder_path):
        logger.error(f"Directory {folder_path} does not exist")
        return []
    
    try:
        if selected_file:
            # Load single selected file
            if os.path.exists(selected_file):
                try:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                    logger.info(f"Loaded {len(all_data)} posts from {selected_file}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from {selected_file}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error loading file {selected_file}: {str(e)}")
            else:
                logger.error(f"Selected file does not exist: {selected_file}")
            return all_data

        # If no file selected, load all files
        timestamped_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        logger.info(f"Found directories: {timestamped_dirs}")
        
        for dir_name in timestamped_dirs:
            # Extract account name from directory name
            account_name = dir_name.split('_')[0]
            if account_name == 'blu':  # Handle special case for blu_es
                account_name = 'blu_es'
            
            json_file = f"{account_name}.json"  # Construct JSON filename
            if account_name == 'blu_es':
                json_file = 'blu_es_.json'
                
            json_path = os.path.join(folder_path, dir_name, json_file)
            
            if os.path.exists(json_path):
                logger.info(f"Processing file: {json_path}")
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Add account name to each post
                        if isinstance(data, list):
                            for post in data:
                                post['account_name'] = account_name
                                post['platform'] = platform
                            all_data.extend(data)
                        else:
                            data['account_name'] = account_name
                            data['platform'] = platform
                            all_data.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from {json_path}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing file {json_path}: {str(e)}")
            else:
                logger.warning(f"File not found: {json_path}")
    except Exception as e:
        logger.error(f"Error processing directory {folder_path}: {str(e)}")
    
    logger.info(f"Loaded {len(all_data)} posts from {platform}")
    return all_data

@lru_cache(maxsize=1024)
def analyze_sentiment(text):
    """Calculate sentiment polarity of text with caching for performance. Returns value between 0-100."""
    if not text:
        return 50  # Neutral sentiment
    blob = TextBlob(str(text))
    # Convert from -1:1 range to 0:100 range
    return (blob.sentiment.polarity + 1) * 50

def analyze_engagement(post):
    """Calculate engagement score based on platform-specific metrics. Returns value between 0-100."""
    if not post:
        return 0
        
    if post.get('platform') == 'instagram':
        likes = post.get('likesCount', 0)
        comments = post.get('commentsCount', 0)
        views = post.get('videoViewCount', 0)
        
        if views > 0:
            # Video content - views are important
            engagement_score = ((likes + comments * 2 + views) / 1000) * 100
        else:
            # Photo content
            engagement_score = ((likes + comments * 2) / 1000) * 100
    else:  # YouTube or other platforms
        likes = post.get('likesCount', post.get('likeCount', 0))
        comments = post.get('commentsCount', post.get('commentCount', 0))
        views = post.get('viewCount', 0)
        
        # Weight engagement differently for other platforms
        engagement_score = ((likes + comments * 2 + views * 0.01) / 1000) * 100
    
    return min(100, max(0, engagement_score))  # Ensure score is between 0-100

def categorize_comment_type(text):
    """Categorize comments into different types."""
    if not text:
        return "no comment"
    
    text = text.lower()
    
    # Check for emojis - positive first, then negative
    if any(emoji in text for emoji in COMMENT_EMOJIS['positive']):
        return "emoji positive"
    elif any(emoji in text for emoji in COMMENT_EMOJIS['negative']):
        return "emoji negative"
    
    # Check simple text characteristics
    if len(text.split()) <= 3:
        return "short response"
    elif '?' in text:
        return "question"
    
    # Check for sentiment keywords
    if any(word in text for word in COMMENT_KEYWORDS['enthusiastic']):
        return "enthusiastic"
    elif any(word in text for word in COMMENT_KEYWORDS['critical']):
        return "critical"
    
    # Default category
    return "general comment"

def process_instagram_data(data):
    """Process raw Instagram data into a structured DataFrame."""
    if not data:
        logger.warning("No data to process")
        return pd.DataFrame()
        
    processed_data = []
    
    for post in data:
        try:
            # Process all comments for the post
            comments = post.get('comments', [])
            comment_sentiments = []
            comment_types = []
            
            # Get the first comment if available for high-level analysis
            first_comment = comments[0].get('text', '') if comments else ''
            first_comment_sentiment = analyze_sentiment(first_comment)
            
            # Process all comments for detailed analysis
            for comment in comments:
                comment_text = comment.get('text', '')
                if comment_text:
                    sentiment = analyze_sentiment(comment_text)
                    comment_type = categorize_comment_type(comment_text)
                    comment_sentiments.append(sentiment)
                    comment_types.append(comment_type)
            
            # Debug logging for comment_sentiments
            logger.debug(f"Comment sentiments: {comment_sentiments}")
            logger.debug(f"Type of comment_sentiments: {type(comment_sentiments)}")
            logger.debug(f"First few comment_sentiments: {comment_sentiments[:5] if len(comment_sentiments) > 5 else comment_sentiments}")
            
            # Calculate comment sentiment stats safely
            avg_comment_sentiment = (
                sum(comment_sentiments) / len(comment_sentiments) 
                if comment_sentiments else 0
            )
            sentiment_variance = (
                np.var(comment_sentiments) 
                if len(comment_sentiments) > 1 else 0
            )
            
            # Count comment types
            type_counts = Counter(comment_types)
            total_comments = len(comment_types) if comment_types else 1  # Avoid division by zero
            
            # Parse timestamp safely
            timestamp = safe_parse_timestamp(
                post.get('timestamp', post.get('createdAt', '')),
                default=datetime.now()
            )
            
            # Calculate engagement score
            engagement_score = analyze_engagement(post)
            
            # Prepare type ratios with safe division
            def safe_ratio(count, total):
                return count / total if total > 0 else 0
            
            row_data = {
                'post_id': post.get('id', ''),
                'timestamp': timestamp,
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday(),  # 0 = Monday, 6 = Sunday
                'likes_count': post.get('likesCount', 0),
                'comments_count': post.get('commentsCount', 0),
                'caption': post.get('caption', ''),
                'caption_sentiment': analyze_sentiment(post.get('caption', '')),
                'caption_length': len(post.get('caption', '')),
                'first_comment': first_comment,
                'first_comment_sentiment': first_comment_sentiment,
                'avg_comment_sentiment': avg_comment_sentiment,
                'sentiment_variance': sentiment_variance,
                'emoji_positive_ratio': safe_ratio(type_counts.get('emoji positive', 0), total_comments),
                'emoji_negative_ratio': safe_ratio(type_counts.get('emoji negative', 0), total_comments),
                'enthusiastic_ratio': safe_ratio(type_counts.get('enthusiastic', 0), total_comments),
                'critical_ratio': safe_ratio(type_counts.get('critical', 0), total_comments),
                'question_ratio': safe_ratio(type_counts.get('question', 0), total_comments),
                'short_response_ratio': safe_ratio(type_counts.get('short response', 0), total_comments),
                'general_comment_ratio': safe_ratio(type_counts.get('general comment', 0), total_comments),
                'engagement_score': engagement_score,
                'type': post.get('type', 'Unknown'),
                'platform': post.get('platform', 'instagram'),
                'account_name': post.get('account_name', '')
            }
            
            processed_data.append(row_data)
            
        except Exception as e:
            logger.error(f"Error processing post {post.get('id', 'unknown')}: {str(e)}")
            continue
    
    # Convert to DataFrame and handle missing values
    df = pd.DataFrame(processed_data)
    
    if df.empty:
        logger.warning("Processed data is empty")
        return df
          # Fill missing values
    df = df.fillna({
        'caption': '',
        'first_comment': '',
        'caption_sentiment': 0,
        'avg_comment_sentiment': 0,
        'engagement_score': 0,
        'type': 'Unknown'
    })
    
    # Add day_name for better readability
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = df['day_of_week'].apply(lambda x: day_names[x])
    
    # Enhance data with additional features
    df = enhance_data_processing(df)
    
    return df

def enhance_data_processing(df):
    """Enhance data processing with additional metrics and features."""
    if df.empty:
        return df
        
    # Add time-based features
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Add content-based features
    df['has_emoji'] = df['caption'].apply(lambda x: any(emoji in x for emoji in COMMENT_EMOJIS['positive'] + COMMENT_EMOJIS['negative']))
    df['hashtag_count'] = df['caption'].str.count('#')
    df['mention_count'] = df['caption'].str.count('@')
    
    # Calculate engagement rates
    df['engagement_rate'] = df.apply(
        lambda row: (row['likes_count'] + row['comments_count']) / 
        (row['reach'] if 'reach' in df.columns and row['reach'] > 0 else 1000), 
        axis=1
    )
    
    # Add sentiment intensity
    df['sentiment_intensity'] = df['caption_sentiment'].abs()
    df['comment_sentiment_intensity'] = df['avg_comment_sentiment'].abs()
    
    # Calculate content quality metrics
    df['caption_word_count'] = df['caption'].str.split().str.len()
    df['avg_comment_length'] = df.apply(
        lambda row: np.mean([len(str(c.get('text', '')).split()) for c in row.get('comments', [])]) 
        if row.get('comments') else 0,
        axis=1
    )
    
    # Add interaction quality metrics
    df['meaningful_comments'] = df.apply(
        lambda row: sum(1 for c in row.get('comments', []) 
                       if len(str(c.get('text', '')).split()) > 3),
        axis=1
    )
    df['meaningful_comment_ratio'] = df['meaningful_comments'] / df['comments_count'].replace(0, 1)
    
    return df

def train_engagement_model(df):
    """Train a machine learning model to predict engagement."""
    if df.empty or len(df) < 10:
        logger.warning("Not enough data to train model")
        return None, None
          # Prepare features for model training
    features = [
        # Content features
        'caption_sentiment', 'caption_length', 'caption_word_count',
        'hashtag_count', 'mention_count', 'has_emoji',
        
        # Engagement features
        'avg_comment_sentiment', 'sentiment_variance', 
        'engagement_rate', 'meaningful_comment_ratio',
        
        # Comment type features
        'emoji_positive_ratio', 'emoji_negative_ratio',
        'enthusiastic_ratio', 'critical_ratio', 'question_ratio',
        'short_response_ratio', 'general_comment_ratio',
        
        # Time features
        'hour_of_day', 'day_of_week', 'is_weekend',
        
        # Sentiment intensity
        'sentiment_intensity', 'comment_sentiment_intensity',
        
        # Quality metrics
        'avg_comment_length'
    ]
    
    # Use only features that exist in the DataFrame
    features = [f for f in features if f in df.columns]
    
    # Ensure we have at least some features
    if not features:
        logger.error("No valid features found for model training")
        return None, None
    
    X = df[features]
    y = df['engagement_score']
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Standardize features (important for tree-based models too)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data with stratification if possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    except ValueError as e:
        logger.warning(f"Could not stratify split: {str(e)}")
        # Fallback to regular split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    
    # Train model with more robust parameters
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate evaluation metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model R² - Train: {train_score:.4f}, Test: {test_score:.4f}")
    
    # Save model and scaler
    ensure_dir_exists(os.path.dirname(MODEL_PATH))
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'features': features,
        'metrics': {
            'train_score': train_score,
            'test_score': test_score
        }
    }, MODEL_PATH)
      # Calculate feature importance
    try:
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        feature_importance = None
    
    if feature_importance is not None and not feature_importance.empty:
        # Clean up feature names for better display
        feature_importance['Feature'] = feature_importance['Feature'].apply(
            lambda x: x.replace('_', ' ').title()
        )
    
    return model, feature_importance

def analyze_comment_patterns(comments):
    """Analyze patterns in comments."""
    if not comments:
        return []
        
    patterns = Counter(comments)
    total = len(comments)
    insights = []
    
    for category, count in patterns.most_common():
        percentage = (count / total) * 100
        if percentage > 5:  # Only report significant patterns
            insights.append(f"- {category.title()}: {percentage:.1f}% of comments")
    
    return insights

def generate_engagement_story(df, account_name=None):
    """Generate insights about engagement timing."""
    if df.empty:
        return "No data available for analysis."
        
    stories = []
    
    if account_name:
        stories.append(f"\n🎯 Account Analysis: @{account_name}")
    
    # Analyze peak engagement times
    hourly_engagement = df.groupby('hour')['engagement_score'].mean()
    if not hourly_engagement.empty:
        peak_hours = hourly_engagement.nlargest(3)
        
        stories.append("\n⏰ Peak Engagement Times:")
        for hour, score in peak_hours.items():
            time_str = f"{hour:02d}:00"
            stories.append(f"- {time_str} UTC with average engagement score of {score:.2f}")
    
    # Day of week analysis
    if 'day_name' in df.columns and df['day_name'].nunique() > 1:
        day_engagement = df.groupby('day_name')['engagement_score'].mean().sort_values(ascending=False)
        best_day = day_engagement.index[0]
        best_score = day_engagement.iloc[0]
        stories.append(f"- Best day of the week: {best_day} (score: {best_score:.2f})")
    
    # Content type impact if available
    if 'type' in df.columns and df['type'].nunique() > 1:
        content_impact = df.groupby('type')['engagement_score'].agg(['mean', 'count'])
        content_impact = content_impact[content_impact['count'] >= 2]  # Only types with 2+ posts
        
        if not content_impact.empty:
            stories.append("\n📊 Content Type Performance:")
            for content_type, stats in content_impact.iterrows():
                stories.append(f"- {content_type} posts ({int(stats['count'])} posts) average {stats['mean']:.2f} engagement points")
    
    # Caption length analysis
    if 'caption_length' in df.columns:
        # Create length buckets
        df['caption_length_bucket'] = pd.cut(
            df['caption_length'], 
            bins=[0, 50, 150, 300, float('inf')],
            labels=['Very Short', 'Short', 'Medium', 'Long']
        )
        length_impact = df.groupby('caption_length_bucket')['engagement_score'].mean().sort_values(ascending=False)
        
        if not length_impact.empty:
            best_length = length_impact.index[0]
            stories.append(f"\n📝 Caption Length Impact:")
            stories.append(f"- {best_length} captions perform best with {length_impact.iloc[0]:.2f} engagement score")
    
    # Sentiment correlation with engagement
    sentiment_corr = df['caption_sentiment'].corr(df['engagement_score'])
    comment_sent_corr = df['avg_comment_sentiment'].corr(df['engagement_score'])
    
    stories.append("\n😊 Sentiment Impact:")
    if abs(sentiment_corr) > 0.2:
        direction = "positive" if sentiment_corr > 0 else "negative"
        stories.append(f"- Posts with {direction} captions tend to get {abs(sentiment_corr):.2f}x more engagement")
    
    if abs(comment_sent_corr) > 0.2:
        direction = "positive" if comment_sent_corr > 0 else "negative"
        stories.append(f"- Posts that attract {direction} comments show {abs(comment_sent_corr):.2f}x more engagement")
    
    return "\n".join(stories)

def generate_natural_language_insights(df, account_name=None):
    """Generate comprehensive insights from the data."""
    if df.empty:
        return ["No data available for analysis."]
        
    insights = []
    
    # Overall sentiment analysis
    avg_caption_sentiment = df['caption_sentiment'].mean()
    avg_comment_sentiment = df['avg_comment_sentiment'].mean()
    
    caption_sentiment_text = (
        "very positive" if avg_caption_sentiment >= SENTIMENT_THRESHOLDS['very_positive']
        else "positive" if avg_caption_sentiment >= SENTIMENT_THRESHOLDS['positive']
        else "neutral" if avg_caption_sentiment >= SENTIMENT_THRESHOLDS['neutral']
        else "negative" if avg_caption_sentiment >= SENTIMENT_THRESHOLDS['negative']
        else "very negative"
    )
    
    comment_sentiment_text = (
        "very positive" if avg_comment_sentiment >= SENTIMENT_THRESHOLDS['very_positive']
        else "positive" if avg_comment_sentiment >= SENTIMENT_THRESHOLDS['positive']
        else "neutral" if avg_comment_sentiment >= SENTIMENT_THRESHOLDS['neutral']
        else "negative" if avg_comment_sentiment >= SENTIMENT_THRESHOLDS['negative']
        else "very negative"
    )
    
    insights.append(f"📝 Content Analysis:")
    insights.append(f"- Your captions tend to be {caption_sentiment_text} (score: {avg_caption_sentiment:.1f}/100)")
    insights.append(f"- Your audience responds with primarily {comment_sentiment_text} comments (score: {avg_comment_sentiment:.1f}/100)")
    
    # Post volume analysis
    post_count = len(df)
    if 'timestamp' in df.columns:
        date_range = pd.to_datetime(df['timestamp'])
        if not date_range.empty:
            date_min = date_range.min()
            date_max = date_range.max()
            days_span = (date_max - date_min).days + 1
            
            if days_span > 0:
                posts_per_day = post_count / days_span
                insights.append(f"- Posting frequency: {posts_per_day:.1f} posts per day over {days_span} days")
    
    # Comment type analysis
    comment_type_columns = [
        'emoji_positive_ratio', 'emoji_negative_ratio', 'enthusiastic_ratio', 
        'critical_ratio', 'question_ratio', 'short_response_ratio', 'general_comment_ratio'
    ]
    
    # Only use columns that exist in the DataFrame
    available_columns = [col for col in comment_type_columns if col in df.columns]
    
    if available_columns:
        comment_type_means = df[available_columns].mean()
        top_type = comment_type_means.idxmax()
        top_ratio = comment_type_means.max() * 100
        
        type_name = top_type.replace('_ratio', '').replace('_', ' ')
        insights.append(f"\n👥 Audience Interaction:")
        insights.append(f"- Most common response type: {type_name} ({top_ratio:.1f}% of comments)")
        
        # Engagement correlation analysis
        highest_correlation = 0
        highest_corr_type = None
        
        for col in available_columns:
            corr = df[col].corr(df['engagement_score'])
            if abs(corr) > abs(highest_correlation):
                highest_correlation = corr
                highest_corr_type = col
        
        if highest_corr_type and abs(highest_correlation) > 0.2:
            type_name = highest_corr_type.replace('_ratio', '').replace('_', ' ')
            corr_direction = "positively" if highest_correlation > 0 else "negatively"
            insights.append(f"- {type_name.title()} comments {corr_direction} correlate with engagement (r={highest_correlation:.2f})")
    
    # Add engagement timing story
    insights.append(generate_engagement_story(df, account_name))
    
    # Add recommendations
    insights.append("\n💡 Recommendations:")
    
    # Time-based recommendation
    if 'hour' in df.columns:
        best_hour = df.groupby('hour')['engagement_score'].mean().idxmax()
        insights.append(f"- Try posting around {best_hour:02d}:00 UTC for maximum engagement")
    
    # Day-based recommendation
    if 'day_name' in df.columns:
        best_day = df.groupby('day_name')['engagement_score'].mean().idxmax()
        insights.append(f"- Consider posting on {best_day}s when engagement tends to be higher")
    
    # Content sentiment recommendation
    if avg_caption_sentiment > 0 and df['caption_sentiment'].corr(df['engagement_score']) > 0.2:
        insights.append("- Keep your positive tone in captions, it resonates well with your audience")
    elif avg_caption_sentiment < 0 and df['caption_sentiment'].corr(df['engagement_score']) > 0.2:
        insights.append("- Your audience responds well to serious content - consider keeping this approach")
    else:
        insights.append("- Experiment with more emotive captions to see if engagement increases")
    
    # Caption length recommendation
    if 'caption_length' in df.columns:
        caption_length_corr = df['caption_length'].corr(df['engagement_score'])
        if abs(caption_length_corr) > 0.2:
            if caption_length_corr > 0:
                insights.append("- Longer captions seem to perform better - consider adding more detail to your posts")
            else:
                insights.append("- Shorter captions tend to perform better - try being more concise")
    
    return insights

def create_visualization_plots(df, account_name=None):
    """Create comprehensive visualization plots."""
    if df.empty:
        return "<div class='alert alert-warning'>No data available for visualization</div>"
    
    # Handle NaN values
    df = df.fillna(0)  # Replace NaN with 0 for numeric columns
    
    title_prefix = f"@{account_name} - " if account_name else ""
    
    # Create subplots layout - more space for better readability
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Sentiment Distribution (0-100 Scale)', 
            'Comment Types', 
            'Engagement vs Sentiment (0-100 Scale)', 
            'Engagement by Hour (0-100 Scale)',
            'Engagement by Day of Week (0-100 Scale)',
            'Caption Length vs Engagement (0-100 Scale)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Update axis ranges and labels to reflect 0-100 scale
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_yaxes(range=[0, 100], row=i, col=j)
            
    # Add a note about the scale
    fig.add_annotation(
        text="All metrics are shown on a 0-100 scale for easy comparison",
        xref="paper", yref="paper",
        x=0, y=1.1,
        showarrow=False,
        font=dict(size=12, color="gray"),
        align="left"
    )
    
    # 1. Sentiment Distribution
    sentiment_hist = go.Histogram(
        x=df['avg_comment_sentiment'], 
        name='Comment Sentiment',
        nbinsx=20,
        marker_color='skyblue'
    )
    fig.add_trace(sentiment_hist, row=1, col=1)
    
    # 2. Comment Types Distribution
    type_columns = [
        'emoji_positive_ratio', 'emoji_negative_ratio', 
        'enthusiastic_ratio', 'critical_ratio', 'question_ratio'
    ]
    
    # Only use columns that exist in the DataFrame
    available_columns = [col for col in type_columns if col in df.columns]
    
    if available_columns:
        type_means = df[available_columns].mean() * 100  # Convert to percentages
        
        # Format labels for readability
        type_labels = [col.replace('_ratio', '').replace('_', ' ').title() for col in type_means.index]
        
        comment_types = go.Bar(
            x=type_labels, 
            y=type_means.values,
            name='Comment Types',
            marker_color='lightgreen'
        )
        fig.add_trace(comment_types, row=1, col=2)
    
    # 3. Engagement vs Sentiment Scatter
    if 'caption_sentiment' in df.columns and 'avg_comment_sentiment' in df.columns:
        engagement_scatter = go.Scatter(
            x=df['caption_sentiment'], 
            y=df['engagement_score'],
            mode='markers',
            name='Engagement vs Caption Sentiment',
            marker=dict(
                size=10,
                color=df['avg_comment_sentiment'],
                colorscale='RdBu',
                colorbar=dict(title='Comment Sentiment'),
                showscale=True,
                opacity=0.7
                ),
            hovertemplate='<b>Caption Sentiment:</b> %{x:.2f}<br>' +
                         '<b>Engagement Score:</b> %{y:.2f}<br>' +
                         '<b>Comment Sentiment:</b> %{marker.color:.2f}'
        )
        fig.add_trace(engagement_scatter, row=2, col=1)
    
    # 4. Engagement by Hour
    if 'hour' in df.columns:
        hourly_avg = df.groupby('hour')['engagement_score'].mean().reset_index()
        hour_bar = go.Bar(
            x=hourly_avg['hour'], 
            y=hourly_avg['engagement_score'],
            name='Engagement by Hour',
            marker_color='coral'
        )
        fig.add_trace(hour_bar, row=2, col=2)
        fig.update_xaxes(title_text='Hour of Day (UTC)', row=2, col=2)
    
    # 5. Engagement by Day of Week
    if 'day_name' in df.columns:
        day_avg = df.groupby('day_name')['engagement_score'].mean().reset_index()
        # Sort by day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_avg['day_order'] = day_avg['day_name'].apply(lambda x: day_order.index(x))
        day_avg = day_avg.sort_values('day_order')
        
        day_bar = go.Bar(
            x=day_avg['day_name'], 
            y=day_avg['engagement_score'],
            name='Engagement by Day',
            marker_color='mediumseagreen'
        )
        fig.add_trace(day_bar, row=3, col=1)
    
    # 6. Caption Length vs Engagement
    if 'caption_length' in df.columns:
        # Create length buckets for better visualization
        df['caption_length_bucket'] = pd.cut(
            df['caption_length'], 
            bins=[0, 50, 150, 300, float('inf')],
            labels=['Very Short', 'Short', 'Medium', 'Long']
        )
        length_avg = df.groupby('caption_length_bucket')['engagement_score'].mean().reset_index()
        
        length_bar = go.Bar(
            x=length_avg['caption_length_bucket'], 
            y=length_avg['engagement_score'],
            name='Caption Length vs Engagement',
            marker_color='darkorchid'
        )
        fig.add_trace(length_bar, row=3, col=2)    # Update layout for better appearance
    fig.update_layout(
        title=f"{title_prefix}Content Analytics Dashboard",
        height=900,
        autosize=True,
        showlegend=False,
        template='plotly_white',
        margin=dict(l=50, r=50, t=100, b=50),
        hovermode='closest',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        ),
        modebar=dict(
            bgcolor='rgba(0,0,0,0)',
            color='#833AB4',
            activecolor='#FD1D1D'
        )
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Percentage (%)', row=1, col=2)
    fig.update_yaxes(title_text='Engagement Score', row=2, col=1)
    fig.update_yaxes(title_text='Engagement Score', row=2, col=2)
    fig.update_yaxes(title_text='Engagement Score', row=3, col=1)
    fig.update_yaxes(title_text='Engagement Score', row=3, col=2)
    
    # Update x-axis titles
    fig.update_xaxes(title_text='Comment Sentiment', row=1, col=1)
    fig.update_xaxes(title_text='Comment Type', row=1, col=2)
    fig.update_xaxes(title_text='Caption Sentiment', row=2, col=1)
    fig.update_xaxes(title_text='Day of Week', row=3, col=1)
    fig.update_xaxes(title_text='Caption Length', row=3, col=2)
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_engagement_heatmap(df):
    """Create a heatmap showing engagement by hour and day of week."""
    if df.empty or 'hour' not in df.columns or 'day_name' not in df.columns:
        return "<div class='alert alert-warning'>Not enough data for heatmap visualization</div>"
    
    # Ensure all hours (0-23) are present in the data
    all_hours = list(range(24))
    if not all(hour in df['hour'].unique() for hour in all_hours):
        temp_data = []
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            for hour in all_hours:
                temp_data.append({
                    'day_name': day,
                    'hour': hour,
                    'engagement_score': 0
                })
        temp_df = pd.DataFrame(temp_data)
        df = pd.concat([df, temp_df], ignore_index=True)
    
    # Prepare data - calculate average engagement by day and hour
    pivot_data = df.pivot_table(
        values='engagement_score',
        index='day_name',
        columns='hour',
        aggfunc='mean',
        fill_value=0
    )
    
    # Debug logging for pivot_data
    logger.debug(f"Pivot data for heatmap: {pivot_data}")
    logger.debug(f"Type of pivot_data: {type(pivot_data)}")
    
    # Ensure proper day ordering
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(day_order)
      # Create heatmap with proper dimensions
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Hour of Day (UTC)", y="Day of Week", color="Engagement Score"),
        x=[f"{h:02d}:00" for h in range(24)],
        y=day_order,
        color_continuous_scale="Viridis",
        title="Engagement Heatmap by Day and Hour",
        aspect='auto'
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate="<b>Day:</b> %{y}<br>" +
                     "<b>Time:</b> %{x}<br>" +
                     "<b>Engagement Score:</b> %{z:.2f}<extra></extra>"
    )
      # Update layout with proper dimensions
    fig.update_layout(
        height=500,
        autosize=True,
        xaxis_title="Hour of Day (UTC)",
        yaxis_title="Day of Week",
        template='plotly_white',
        margin=dict(l=50, r=50, t=100, b=50),
        coloraxis_colorbar=dict(
            title="Engagement Score",
            thicknessmode="pixels",
            thickness=20,
            lenmode="pixels",
            len=400
        )
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def visualize_feature_importance(feature_importance):
    """Create a horizontal bar chart of feature importance."""
    if feature_importance is None or feature_importance.empty:
        return "<div class='alert alert-warning'>No feature importance data available</div>"
    
    # Debug logging for feature_importance
    logger.debug(f"Feature importance data: {feature_importance}")
    logger.debug(f"Type of feature_importance: {type(feature_importance)}")
    
    # Get top features
    top_features = feature_importance.sort_values('Importance', ascending=True).tail(10)
    
    # Create horizontal bar chart
    fig = px.bar(
        top_features,
        y='Feature',
        x='Importance',
        orientation='h',
        title='Top Factors Influencing Engagement',
        labels={'Importance': 'Relative Importance', 'Feature': ''},
        color='Importance',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(
        height=500,
        autosize=True,
        template='plotly_white',
        margin=dict(l=150, r=50, t=100, b=50),  # Increased left margin for feature names
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=12)
        )
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def safe_correlation(x, y):
    """Calculate correlation while handling NaN values."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if not np.any(mask) or len(x[mask]) < 2:
        return 0
    return np.corrcoef(x[mask], y[mask])[0, 1]

def process_correlations(df):
    """Process correlations with proper NaN handling."""
    correlations = {}
    for col in ['caption_sentiment', 'avg_comment_sentiment', 'caption_length']:
        if col in df.columns:
            correlations[col] = safe_correlation(
                df[col].values, 
                df['engagement_score'].values
            )
    return correlations

def normalize_sentiment_score(score):
    """Convert sentiment score from -1 to 1 range to 0 to 100 range."""
    return (score + 1) * 50  # This converts -1 to 1 into 0 to 100

def get_sentiment_class(score):
    """Get sentiment class based on normalized score (0-100 range)."""
    normalized_score = normalize_sentiment_score(score)
    
    if normalized_score >= 80:
        return ('Very Positive', 'text-success')
    elif normalized_score >= 60:
        return ('Positive', 'text-success')
    elif normalized_score >= 40:
        return ('Neutral', 'text-muted')
    elif normalized_score >= 20:
        return ('Negative', 'text-danger')
    else:
        return ('Very Negative', 'text-danger')

# Flask routes
@app.route('/')
def index():
    """Main page that lists available data files."""
    available_files = get_available_data_files()
    return render_template('index.html', available_files=available_files)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze selected data file and display results."""
    selected_file = request.form.get('selected_file')
    
    if not selected_file:
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        })
    
    try:
        # Extract account name and load data
        account_name = os.path.basename(os.path.dirname(selected_file)).split('_')[0]
        if account_name == 'blu':
            account_name = 'blu_es'
        
        raw_data = load_social_media_data(selected_file=selected_file)
        if not raw_data:
            return jsonify({
                'status': 'error',
                'message': 'Failed to load data from file'
            })
        
        processed_df = process_instagram_data(raw_data)
        if processed_df.empty:
            return jsonify({
                'status': 'error',
                'message': 'No data available in selected file'
            })
        
        # Train model and get feature importance
        model, feature_importance = train_engagement_model(processed_df)
        
        # Generate insights
        insights = generate_natural_language_insights(processed_df, account_name)
        
        # Prepare sentiment analysis data with scale information
        avg_caption_sentiment = processed_df['caption_sentiment'].mean()
        avg_comment_sentiment = processed_df['avg_comment_sentiment'].mean()
        
        content_tone, content_class = get_sentiment_class(avg_caption_sentiment)
        audience_tone, audience_class = get_sentiment_class(avg_comment_sentiment)
        
        sentiment_data = {
            'content_summary': f"""
                <p class="mb-2">Overall Tone: <span class="{content_class} fw-bold">{content_tone}</span></p>
                <p class="mb-0">Your content generally maintains a {content_tone.lower()} tone 
                with a sentiment score of {avg_caption_sentiment:.1f}/100</p>
            """,
            'audience_summary': f"""
                <p class="mb-2">Overall Response: <span class="{audience_class} fw-bold">{audience_tone}</span></p>
                <p class="mb-0">Your audience typically responds with {audience_tone.lower()} comments 
                with an average sentiment of {avg_comment_sentiment:.1f}/100</p>
            """
        }

        # Prepare engagement visualization data
        hourly_engagement = processed_df.groupby('hour')['engagement_score'].mean()
        engagement_data = {
            'type': 'line',
            'data': {
                'labels': [f"{h:02d}:00" for h in range(24)],
                'datasets': [{
                    'label': 'Average Engagement Score',
                    'data': hourly_engagement.reindex(range(24)).fillna(0).tolist(),
                    'fill': False,
                    'borderColor': 'rgb(75, 192, 192)',
                    'tension': 0.1
                }]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'scales': {
                    'y': {
                        'beginAtZero': True,
                        'title': {
                            'display': True,
                            'text': 'Engagement Score'
                        }
                    },
                    'x': {
                        'title': {
                            'display': True,
                            'text': 'Hour of Day (UTC)'
                        }
                    }
                },
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'Engagement Throughout the Day'
                    }
                }
            }
        }

        # Prepare feature importance visualization
        if feature_importance is not None and not feature_importance.empty:
            top_features = feature_importance.nlargest(10, 'Importance')
            feature_viz = {
                'type': 'bar',
                'data': {
                    'labels': top_features['Feature'].tolist(),
                    'datasets': [{
                        'label': 'Feature Importance',
                        'data': top_features['Importance'].tolist(),
                        'backgroundColor': 'rgba(153, 102, 255, 0.2)',
                        'borderColor': 'rgba(153, 102, 255, 1)',
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'indexAxis': 'y',
                    'scales': {
                        'x': {
                            'beginAtZero': True,
                            'title': {
                                'display': True,
                                'text': 'Importance Score'
                            }
                        }
                    },
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Factors Influencing Engagement'
                        }
                    }
                }
            }
        else:
            feature_viz = None

        response_data = {
            'status': 'success',
            'account_name': account_name,
            'post_count': len(processed_df),
            'insights': insights,
            'sentiment': sentiment_data,
            'visualizations': engagement_data,
            'feature_importance': feature_viz
        }
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error analyzing file {selected_file}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error analyzing data: {str(e)}'
        })

@app.route('/get_data_files')
def get_data_files():
    """API endpoint to get available data files."""
    available_files = get_available_data_files()
    return jsonify(available_files)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get content recommendations based on past performance."""
    selected_file = request.form.get('selected_file')
    
    if not selected_file:
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        })
    
    try:
        # Load and process data
        raw_data = load_social_media_data(selected_file=selected_file)
        processed_df = process_instagram_data(raw_data)
        
        if processed_df.empty:
            return jsonify({
                'status': 'error',
                'message': 'Not enough data available for recommendations'
            })
        
        # Best time to post
        best_hour = processed_df.groupby('hour')['engagement_score'].mean().idxmax()
        best_day = processed_df.groupby('day_name')['engagement_score'].mean().idxmax()
        
        # Best content type if available
        best_content_type = None
        if 'type' in processed_df.columns and processed_df['type'].nunique() > 1:
            best_content_type = processed_df.groupby('type')['engagement_score'].mean().idxmax()
        
        # Optimal caption attributes
        avg_caption_sentiment = processed_df['caption_sentiment'].mean()
        sentiment_direction = 'positive' if avg_caption_sentiment > 0 else 'negative' if avg_caption_sentiment < 0 else 'neutral'
        
        # Optimal caption length
        if 'caption_length' in processed_df.columns:
            processed_df['caption_length_bucket'] = pd.cut(
                processed_df['caption_length'], 
                bins=[0, 50, 150, 300, float('inf')],
                labels=['Very Short', 'Short', 'Medium', 'Long']
            )
            best_length = processed_df.groupby('caption_length_bucket')['engagement_score'].mean().idxmax()
        else:
            best_length = 'Unknown'
        
        return jsonify({
            'status': 'success',
            'best_time': {
                'hour': f"{best_hour:02d}:00 UTC",
                'day': best_day
            },
            'content_preferences': {
                'best_type': best_content_type,
                'sentiment': sentiment_direction,
                'caption_length': best_length
            }
        })
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error generating recommendations: {str(e)}'
        })

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Make sure the data folders exist
    ensure_dir_exists(DATA_FOLDER)
    ensure_dir_exists(os.path.dirname(MODEL_PATH))
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)