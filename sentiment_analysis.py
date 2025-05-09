import json
import os
from textblob import TextBlob
import pandas as pd
from datetime import datetime
import numpy as np
from collections import Counter

def load_social_media_data(folder_path, platform):
    all_data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Add platform identifier
                    if isinstance(data, list):
                        for item in data:
                            item['platform'] = platform
                        all_data.extend(data)
                    else:
                        data['platform'] = platform
                        all_data.append(data)
    return all_data

def analyze_sentiment(text):
    if not text:
        return 0
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

def analyze_engagement(post):
    """Calculate engagement score based on platform-specific metrics"""
    if post['platform'] == 'instagram':
        likes = post.get('likesCount', 0)
        comments = post.get('commentsCount', 0)
        if 'videoViewCount' in post:
            engagement_score = (likes + comments * 2 + post['videoViewCount']) / 100
        else:
            engagement_score = (likes + comments * 2) / 100
    else:  # YouTube
        views = post.get('viewCount', 0)
        likes = post.get('likeCount', 0)
        comments = post.get('commentCount', 0)
        # YouTube engagement formula with higher weight for comments
        engagement_score = (views/1000 + likes * 2 + comments * 3) / 100
    
    return engagement_score

def categorize_comment_type(text):
    if not text:
        return "no comment"
    
    text = text.lower()
    if any(emoji in text for emoji in ['❤️', '😍', '🔥', '👏', '💙', '😊']):
        return "emoji positive"
    elif any(emoji in text for emoji in ['😢', '😭', '💔', '😞', '😪']):
        return "emoji negative"
    elif len(text.split()) <= 3:
        return "short response"
    elif '?' in text:
        return "question"
    elif any(word in text for word in ['amazing', 'beautiful', 'awesome', 'great', 'love']):
        return "enthusiastic"
    elif any(word in text for word in ['fake', 'bad', 'terrible', 'hate', 'worst']):
        return "critical"
    else:
        return "general comment"

def analyze_comment_patterns(comments):
    patterns = Counter(comments)
    total = len(comments)
    insights = []
    
    for category, count in patterns.most_common():
        percentage = (count / total) * 100
        if percentage > 5:  # Only report significant patterns
            insights.append(f"- {category.title()}: {percentage:.1f}% of comments")
    
    return insights

def generate_engagement_story(df, platform):
    stories = []
    
    # Analyze peak engagement times
    df['hour'] = df['timestamp'].dt.hour
    peak_hours = df.groupby('hour')['engagement_score'].mean().nlargest(3)
    
    stories.append(f"\n{platform.title()} Engagement Timing Insights:")
    stories.append(f"Your {platform} audience is most active during:")
    for hour, score in peak_hours.items():
        stories.append(f"- {hour:02d}:00 UTC with {int(score)} engagement points")
    
    # Content type impact
    if 'type' in df.columns:
        content_impact = df.groupby('type')['engagement_score'].agg(['mean', 'count'])
        stories.append(f"\n{platform.title()} Content Performance:")
        for content_type, stats in content_impact.iterrows():
            stories.append(f"- {content_type} posts ({int(stats['count'])} posts) typically get {int(stats['mean'])} engagement points")
    
    return "\n".join(stories)

def generate_platform_insights(df, platform):
    insights = []
    
    # Overall sentiment analysis
    avg_caption_sentiment = df['caption_sentiment'].mean()
    sentiment_direction = "positive" if avg_caption_sentiment > 0 else "negative" if avg_caption_sentiment < 0 else "neutral"
    sentiment_strength = abs(avg_caption_sentiment)
    
    insights.append(f"\n🎯 {platform.title()} Content Style:")
    if sentiment_strength > 0.5:
        insights.append(f"Your {platform} content has a strong {sentiment_direction} vibe that really stands out!")
    elif sentiment_strength > 0.2:
        insights.append(f"You're maintaining a pleasantly {sentiment_direction} tone in your {platform} posts.")
    else:
        insights.append(f"You're keeping things balanced with a neutral approach on {platform}.")
    
    # Audience response
    comment_types = [categorize_comment_type(comment) for comment in df['firstComment'].dropna()]
    comment_patterns = analyze_comment_patterns(comment_types)
    
    insights.append(f"\n👥 {platform.title()} Audience Behavior:")
    insights.extend(comment_patterns)
    
    # Engagement correlation
    sentiment_engagement_corr = df['caption_sentiment'].corr(df['engagement_score'])
    
    insights.append(f"\n💫 {platform.title()} Engagement Patterns:")
    if abs(sentiment_engagement_corr) > 0.3:
        correlation_type = "positive" if sentiment_engagement_corr > 0 else "negative"
        if correlation_type == "positive":
            insights.append(f"Your {platform} audience loves your upbeat content! Keep that positive energy flowing.")
        else:
            insights.append(f"Interestingly, your {platform} followers engage more with serious or thought-provoking content.")
    
    # Add engagement timing story
    insights.append(generate_engagement_story(df, platform))
    
    return "\n".join(insights)

def compare_platforms(insta_df, youtube_df):
    comparison = []
    
    comparison.append("\n🔄 Platform Comparison:")
    
    # Compare average engagement
    insta_eng = insta_df['engagement_score'].mean()
    youtube_eng = youtube_df['engagement_score'].mean()
    
    better_platform = "Instagram" if insta_eng > youtube_eng else "YouTube"
    comparison.append(f"\n📊 Overall Engagement Winner: {better_platform}")
    comparison.append(f"- Instagram average engagement: {int(insta_eng)} points")
    comparison.append(f"- YouTube average engagement: {int(youtube_eng)} points")
    
    # Compare sentiment effectiveness
    insta_sent_corr = insta_df['caption_sentiment'].corr(insta_df['engagement_score'])
    youtube_sent_corr = youtube_df['caption_sentiment'].corr(youtube_df['engagement_score'])
    
    comparison.append("\n😊 Content Tone Impact:")
    comparison.append(f"- Instagram audience: {'Prefers positive' if insta_sent_corr > 0 else 'Engages with serious'} content")
    comparison.append(f"- YouTube audience: {'Prefers positive' if youtube_sent_corr > 0 else 'Engages with serious'} content")
    
    # Peak timing comparison
    insta_peak = insta_df.groupby('hour')['engagement_score'].mean().idxmax()
    youtube_peak = youtube_df.groupby('hour')['engagement_score'].mean().idxmax()
    
    comparison.append("\n⏰ Best Posting Times (UTC):")
    comparison.append(f"- Instagram: {insta_peak:02d}:00")
    comparison.append(f"- YouTube: {youtube_peak:02d}:00")
    
    return "\n".join(comparison)

def main():
    # Paths to data folders
    instagram_folder = 'instagram_data'
    youtube_folder = 'youtube_data'
    target_account = 'blu_es_'  # for Instagram
    target_youtube = 'MrBeast'  # for YouTube
    
    insta_results = []
    youtube_results = []
    
    # Process Instagram data
    insta_data = load_social_media_data(instagram_folder, 'instagram')
    for post in insta_data:
        if 'inputUrl' in post and target_account in post['inputUrl']:
            caption_sentiment = analyze_sentiment(post.get('caption', ''))
            # Get the first comment if available
            first_comment = post.get('comments', [{}])[0].get('text', '') if post.get('comments', []) else ''
            first_comment_sentiment = analyze_sentiment(first_comment)
            engagement_score = analyze_engagement(post)
            try:
                timestamp = datetime.strptime(post.get('timestamp', post.get('createdAt', '')), '%Y-%m-%dT%H:%M:%S.000Z')
            except ValueError:
                try:
                    # Try parsing date from shortcode or use current date as fallback
                    timestamp = datetime.now()
                except:
                    timestamp = datetime.now()
            
            insta_results.append({
                'platform': 'instagram',
                'account': target_account,
                'post_id': post.get('id', ''),
                'timestamp': timestamp,
                'caption_sentiment': caption_sentiment,
                'first_comment_sentiment': first_comment_sentiment,
                'engagement_score': engagement_score,
                'type': post.get('type', 'Unknown'),
                'firstComment': first_comment
            })
    
    # Process YouTube data
    youtube_data = load_social_media_data(youtube_folder, 'youtube')
    for post in youtube_data:
        if 'channel' in post and target_youtube in post['channel']:
            description_sentiment = analyze_sentiment(post.get('description', ''))
            top_comment = post.get('comments', [{}])[0].get('text', '') if post.get('comments', []) else ''
            top_comment_sentiment = analyze_sentiment(top_comment)
            engagement_score = analyze_engagement(post)
            try:
                timestamp = datetime.strptime(post.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%S.000Z')
            except ValueError:
                timestamp = datetime.now()
            
            youtube_results.append({
                'platform': 'youtube',
                'account': post.get('channel', ''),
                'post_id': post.get('id', ''),
                'timestamp': timestamp,
                'caption_sentiment': description_sentiment,
                'first_comment_sentiment': top_comment_sentiment,
                'engagement_score': engagement_score,
                'type': 'video',
                'firstComment': top_comment
            })
    
    # Convert to DataFrames
    insta_df = pd.DataFrame(insta_results)
    youtube_df = pd.DataFrame(youtube_results)
    
    if len(insta_df) == 0 and len(youtube_df) == 0:
        print("No data found for the specified accounts")
        return
    
    print("\n====== Social Media Content Analysis ======")
    
    if len(insta_df) > 0:
        print("\n📱 Instagram Insights for @" + target_account)
        print(generate_platform_insights(insta_df, "instagram"))
        
    if len(youtube_df) > 0:
        print("\n🎥 YouTube Insights for " + target_youtube)
        print(generate_platform_insights(youtube_df, "youtube"))
    
    if len(insta_df) > 0 and len(youtube_df) > 0:
        print("\n🔍 Cross-Platform Analysis")
        print(compare_platforms(insta_df, youtube_df))
    
    # Save detailed results
    if len(insta_df) > 0:
        insta_df.to_csv(f'{target_account}_instagram_analysis.csv', index=False)
    if len(youtube_df) > 0:
        youtube_df.to_csv(f'{target_youtube}_youtube_analysis.csv', index=False)

if __name__ == "__main__":
    main()