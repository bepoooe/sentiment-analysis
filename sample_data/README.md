# Sample Data Structure

This folder contains examples of the expected data structure for both Instagram and YouTube data.

## Instagram Data Structure
Place your Instagram JSON data files in the `instagram_data` folder. Each file should have the following structure:

```json
{
  "inputUrl": "https://www.instagram.com/username/",
  "id": "post_id",
  "type": "Image/Video/Sidecar",
  "caption": "post caption",
  "hashtags": ["tag1", "tag2"],
  "commentsCount": 123,
  "likesCount": 456,
  "timestamp": "2025-05-01T15:40:46.000Z"
}
```

## YouTube Data Structure
Place your YouTube JSON data files in the `youtube_data` folder. Each file should have the following structure:

```json
{
  "channel": "channel_name",
  "videoId": "video_id",
  "description": "video description",
  "viewCount": 1000,
  "likeCount": 100,
  "commentCount": 50,
  "publishedAt": "2025-05-01T12:31:52.000Z"
}
```

## Data Organization
- Each platform should have its own folder
- Data files should be in JSON format
- Timestamps should be in ISO 8601 format
