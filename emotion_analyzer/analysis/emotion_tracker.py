import time
from collections import Counter, defaultdict


class EmotionTracker:
    def __init__(self):
        """Initialize the emotion tracker with empty history"""
        self.emotion_history = []
        self.emotion_counts = Counter()
        self.emotion_timeline = []
    
    def add_emotion_event(self, fused_result):
        """
        Add a fused emotion result to the history
        
        Args:
            fused_result: Dictionary containing fused emotion data
        """
        if fused_result and 'fused_emotion' in fused_result:
            # Add to history
            self.emotion_history.append(fused_result)
            
            # Update counts
            emotion = fused_result['fused_emotion']
            self.emotion_counts[emotion] += 1
            
            # Add to timeline with timestamp
            timestamp = fused_result.get('timestamp', time.time())
            self.emotion_timeline.append({
                'timestamp': timestamp,
                'emotion': emotion,
                'confidence': fused_result.get('confidence', 0.5)
            })
    
    def get_history(self):
        """
        Get the complete emotion history
        
        Returns:
            List of all emotion events
        """
        return self.emotion_history.copy()
    
    def get_emotion_statistics(self):
        """
        Get statistics about the tracked emotions
        
        Returns:
            Dictionary with emotion statistics
        """
        if not self.emotion_history:
            return {
                'total_events': 0,
                'emotion_distribution': {},
                'most_frequent_emotion': 'neutral',
                'emotion_percentages': {}
            }
        
        total_events = len(self.emotion_history)
        
        # Calculate percentages
        emotion_percentages = {}
        for emotion, count in self.emotion_counts.items():
            emotion_percentages[emotion] = (count / total_events) * 100
        
        # Find most frequent emotion
        most_frequent = self.emotion_counts.most_common(1)[0][0] if self.emotion_counts else 'neutral'
        
        return {
            'total_events': total_events,
            'emotion_distribution': dict(self.emotion_counts),
            'most_frequent_emotion': most_frequent,
            'emotion_percentages': emotion_percentages
        }
    
    def get_emotion_timeline(self):
        """
        Get the emotion timeline with timestamps
        
        Returns:
            List of emotion events with timestamps
        """
        return self.emotion_timeline.copy()
    
    def get_emotion_periods(self, emotion, min_duration=5.0):
        """
        Get periods where a specific emotion was dominant
        
        Args:
            emotion: Emotion to search for
            min_duration: Minimum duration in seconds for a period
            
        Returns:
            List of periods (start_time, end_time, duration)
        """
        periods = []
        current_period_start = None
        
        for event in self.emotion_timeline:
            if event['emotion'] == emotion:
                if current_period_start is None:
                    current_period_start = event['timestamp']
            else:
                if current_period_start is not None:
                    duration = event['timestamp'] - current_period_start
                    if duration >= min_duration:
                        periods.append((current_period_start, event['timestamp'], duration))
                    current_period_start = None
        
        # Handle case where period extends to the end
        if current_period_start is not None and self.emotion_timeline:
            last_timestamp = self.emotion_timeline[-1]['timestamp']
            duration = last_timestamp - current_period_start
            if duration >= min_duration:
                periods.append((current_period_start, last_timestamp, duration))
        
        return periods
    
    def clear_history(self):
        """Clear all tracked emotion data"""
        self.emotion_history.clear()
        self.emotion_counts.clear()
        self.emotion_timeline.clear()
