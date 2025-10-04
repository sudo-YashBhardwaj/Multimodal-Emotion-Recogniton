from collections import Counter
import numpy as np


class FusionEngine:
    def __init__(self, config):
        self.config = config
        # Priority order for emotion fusion (higher index = higher priority)
        self.emotion_priority = {
            'text': 3,
            'video': 2,
            'audio': 1
        }
    
    def fuse(self, video_results, audio_results, text_results):
        """
        Fuse results from all modalities using late fusion strategy
        
        Args:
            video_results: List of video emotion detection results
            audio_results: Dictionary with audio emotion results
            text_results: Dictionary with text emotion results
            
        Returns:
            Dictionary with fused emotion state
        """
        # Initialize source emotions
        source_emotions = {
            'video': 'neutral',
            'audio': 'neutral',
            'text': 'neutral'
        }
        
        # Extract emotions from each modality
        video_emotions = self._extract_video_emotions(video_results)
        audio_emotion = self._extract_audio_emotion(audio_results)
        text_emotion = self._extract_text_emotion(text_results)
        
        # Update source emotions
        if video_emotions:
            source_emotions['video'] = video_emotions[0]  # Take the first/most confident
        if audio_emotion:
            source_emotions['audio'] = audio_emotion
        if text_emotion:
            source_emotions['text'] = text_emotion
        
        # Fuse emotions using priority-based approach
        fused_emotion = self._fuse_emotions_by_priority(source_emotions)
        
        # Get timestamp (use the most recent one available)
        timestamp = self._get_timestamp(video_results, audio_results, text_results)
        
        return {
            'timestamp': timestamp,
            'fused_emotion': fused_emotion,
            'source_emotions': source_emotions,
            'confidence': self._calculate_confidence(source_emotions)
        }
    
    def _extract_video_emotions(self, video_results):
        """
        Extract emotions from video results
        
        Args:
            video_results: List of video detection results
            
        Returns:
            List of emotion strings
        """
        emotions = []
        if video_results:
            for result in video_results:
                if 'dominant_emotion' in result:
                    emotions.append(result['dominant_emotion'])
        return emotions
    
    def _extract_audio_emotion(self, audio_results):
        """
        Extract emotion from audio results
        
        Args:
            audio_results: Dictionary with audio results
            
        Returns:
            Emotion string or None
        """
        if audio_results and 'ser_emotion' in audio_results:
            return audio_results['ser_emotion']
        return None
    
    def _extract_text_emotion(self, text_results):
        """
        Extract emotion from text results
        
        Args:
            text_results: Dictionary with text results
            
        Returns:
            Emotion string or None
        """
        if text_results and 'dominant_emotion' in text_results:
            return text_results['dominant_emotion']
        return None
    
    def _fuse_emotions_by_priority(self, source_emotions):
        """
        Fuse emotions using priority-based approach
        
        Args:
            source_emotions: Dictionary with emotions from each modality
            
        Returns:
            Fused emotion string
        """
        # Filter out neutral emotions for better fusion
        non_neutral_emotions = {
            modality: emotion for modality, emotion in source_emotions.items()
            if emotion != 'neutral'
        }
        
        if not non_neutral_emotions:
            return 'neutral'
        
        # If only one non-neutral emotion, use it
        if len(non_neutral_emotions) == 1:
            return list(non_neutral_emotions.values())[0]
        
        # Use priority-based selection
        for modality in ['text', 'video', 'audio']:
            if modality in non_neutral_emotions:
                return non_neutral_emotions[modality]
        
        # Fallback to most common emotion
        emotion_counts = Counter(source_emotions.values())
        return emotion_counts.most_common(1)[0][0]
    
    def _get_timestamp(self, video_results, audio_results, text_results):
        """
        Get the most recent timestamp from all results
        
        Args:
            video_results: Video results
            audio_results: Audio results
            text_results: Text results
            
        Returns:
            Most recent timestamp
        """
        timestamps = []
        
        if video_results:
            for result in video_results:
                if 'timestamp' in result:
                    timestamps.append(result['timestamp'])
        
        if audio_results and 'timestamp' in audio_results:
            timestamps.append(audio_results['timestamp'])
        
        if text_results and 'timestamp' in text_results:
            timestamps.append(text_results['timestamp'])
        
        return max(timestamps) if timestamps else 0.0
    
    def _calculate_confidence(self, source_emotions):
        """
        Calculate confidence score for the fused result
        
        Args:
            source_emotions: Dictionary with emotions from each modality
            
        Returns:
            Confidence score between 0 and 1
        """
        # Count non-neutral emotions
        non_neutral_count = sum(1 for emotion in source_emotions.values() if emotion != 'neutral')
        total_modalities = len(source_emotions)
        
        # Base confidence on agreement between modalities
        if non_neutral_count == 0:
            return 0.5  # Neutral confidence for all neutral
        
        # Check for agreement
        emotion_counts = Counter(source_emotions.values())
        most_common_count = emotion_counts.most_common(1)[0][1]
        
        # Confidence based on agreement
        agreement_ratio = most_common_count / total_modalities
        return min(0.9, 0.5 + (agreement_ratio * 0.4))
