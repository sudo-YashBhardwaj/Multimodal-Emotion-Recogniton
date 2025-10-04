from collections import Counter
import time


class Summarizer:
    def __init__(self):
        """Initialize the summarizer"""
        pass
    
    def generate_summary(self, emotion_history):
        """
        Generate a summary of the emotion analysis session
        
        Args:
            emotion_history: List of emotion events from EmotionTracker
            
        Returns:
            Formatted multi-line string with the summary
        """
        if not emotion_history:
            return "No emotion data available for analysis."
        
        # Analyze emotion distribution
        emotion_counts = Counter()
        timestamps = []
        
        for event in emotion_history:
            if 'fused_emotion' in event:
                emotion_counts[event['fused_emotion']] += 1
            if 'timestamp' in event:
                timestamps.append(event['timestamp'])
        
        # Calculate statistics
        total_events = len(emotion_history)
        most_frequent_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else 'neutral'
        most_frequent_count = emotion_counts[most_frequent_emotion]
        most_frequent_percentage = (most_frequent_count / total_events) * 100
        
        # Find strong emotion periods
        strong_emotions = ['angry', 'sad', 'fear', 'disgust']
        strong_emotion_periods = self._find_strong_emotion_periods(emotion_history, strong_emotions)
        
        # Calculate session duration
        session_duration = 0
        if timestamps:
            session_duration = max(timestamps) - min(timestamps)
        
        # Generate summary text
        summary_lines = [
            "=== Emotion Analysis Summary ===",
            f"Session Duration: {session_duration:.1f} seconds",
            f"Total Emotion Events: {total_events}",
            "",
            "=== Overall Emotion Distribution ===",
        ]
        
        # Add emotion percentages
        for emotion, count in emotion_counts.most_common():
            percentage = (count / total_events) * 100
            summary_lines.append(f"- {emotion.capitalize()}: {count} events ({percentage:.1f}%)")
        
        summary_lines.extend([
            "",
            f"=== Key Findings ===",
            f"- Most frequent emotion: {most_frequent_emotion.capitalize()} ({most_frequent_percentage:.1f}%)",
        ])
        
        # Add strong emotion periods if found
        if strong_emotion_periods:
            summary_lines.append("")
            summary_lines.append("=== Strong Emotion Periods ===")
            for emotion, periods in strong_emotion_periods.items():
                if periods:
                    summary_lines.append(f"- {emotion.capitalize()} periods:")
                    for start_time, end_time, duration in periods:
                        summary_lines.append(f"  * {start_time:.1f}s - {end_time:.1f}s (duration: {duration:.1f}s)")
        else:
            summary_lines.append("- No significant strong emotion periods detected")
        
        # Add overall sentiment assessment
        sentiment_assessment = self._assess_overall_sentiment(emotion_counts)
        summary_lines.extend([
            "",
            "=== Overall Sentiment Assessment ===",
            f"- {sentiment_assessment}"
        ])
        
        return "\n".join(summary_lines)
    
    def _find_strong_emotion_periods(self, emotion_history, strong_emotions, min_duration=3.0):
        """
        Find periods where strong emotions were dominant
        
        Args:
            emotion_history: List of emotion events
            strong_emotions: List of emotions considered "strong"
            min_duration: Minimum duration for a period to be significant
            
        Returns:
            Dictionary mapping emotions to their periods
        """
        periods = {emotion: [] for emotion in strong_emotions}
        
        for emotion in strong_emotions:
            current_period_start = None
            
            for i, event in enumerate(emotion_history):
                if 'fused_emotion' in event and event['fused_emotion'] == emotion:
                    if current_period_start is None:
                        current_period_start = event.get('timestamp', i)
                else:
                    if current_period_start is not None:
                        end_time = event.get('timestamp', i)
                        duration = end_time - current_period_start
                        if duration >= min_duration:
                            periods[emotion].append((current_period_start, end_time, duration))
                        current_period_start = None
            
            # Handle case where period extends to the end
            if current_period_start is not None:
                last_timestamp = emotion_history[-1].get('timestamp', len(emotion_history) - 1)
                duration = last_timestamp - current_period_start
                if duration >= min_duration:
                    periods[emotion].append((current_period_start, last_timestamp, duration))
        
        # Filter out empty periods
        return {emotion: periods for emotion, periods in periods.items() if periods}
    
    def _assess_overall_sentiment(self, emotion_counts):
        """
        Assess the overall sentiment based on emotion distribution
        
        Args:
            emotion_counts: Counter object with emotion frequencies
            
        Returns:
            String describing the overall sentiment
        """
        if not emotion_counts:
            return "Unable to assess sentiment - no emotion data"
        
        # Define sentiment categories
        positive_emotions = ['happy', 'joy', 'excited', 'surprise']
        negative_emotions = ['angry', 'sad', 'fear', 'disgust']
        neutral_emotions = ['neutral', 'calm']
        
        # Count emotions by category
        positive_count = sum(emotion_counts[emotion] for emotion in positive_emotions if emotion in emotion_counts)
        negative_count = sum(emotion_counts[emotion] for emotion in negative_emotions if emotion in emotion_counts)
        neutral_count = sum(emotion_counts[emotion] for emotion in neutral_emotions if emotion in emotion_counts)
        
        total_count = sum(emotion_counts.values())
        
        if total_count == 0:
            return "No emotion data available"
        
        positive_ratio = positive_count / total_count
        negative_ratio = negative_count / total_count
        neutral_ratio = neutral_count / total_count
        
        # Determine overall sentiment
        if positive_ratio > 0.4:
            return f"Positive sentiment (positive: {positive_ratio:.1%}, negative: {negative_ratio:.1%}, neutral: {neutral_ratio:.1%})"
        elif negative_ratio > 0.4:
            return f"Negative sentiment (positive: {positive_ratio:.1%}, negative: {negative_ratio:.1%}, neutral: {neutral_ratio:.1%})"
        else:
            return f"Neutral sentiment (positive: {positive_ratio:.1%}, negative: {negative_ratio:.1%}, neutral: {neutral_ratio:.1%})"
