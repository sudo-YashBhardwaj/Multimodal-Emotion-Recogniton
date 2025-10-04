import cv2
import numpy as np


class DisplayManager:
    def __init__(self):
        """Initialize the display manager"""
        # Define colors for different emotions (BGR format)
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'fear': (0, 255, 255),     # Yellow
            'surprise': (255, 0, 255), # Magenta
            'disgust': (0, 165, 255),  # Orange
            'neutral': (128, 128, 128), # Gray
            'excited': (0, 255, 127),   # Spring Green
            'calm': (255, 255, 0)       # Cyan
        }
        
        # Default color for unknown emotions
        self.default_color = (255, 255, 255)  # White
    
    def draw_results(self, frame, video_results, fused_emotion_result, summary_text=""):
        """
        Draw emotion analysis results on the video frame
        
        Args:
            frame: Original video frame (BGR format)
            video_results: List of video emotion detection results
            fused_emotion_result: Fused emotion result dictionary
            summary_text: Summary text to display
            
        Returns:
            Modified frame with overlays
        """
        if frame is None:
            return frame
        
        # Create a copy of the frame to avoid modifying the original
        display_frame = frame.copy()
        
        # Draw face detection results
        display_frame = self._draw_face_detections(display_frame, video_results)
        
        # Draw fused emotion
        display_frame = self._draw_fused_emotion(display_frame, fused_emotion_result)
        
        # Draw summary text
        if summary_text:
            display_frame = self._draw_summary_text(display_frame, summary_text)
        
        # Draw frame info
        display_frame = self._draw_frame_info(display_frame)
        
        return display_frame
    
    def _draw_face_detections(self, frame, video_results):
        """
        Draw face detection boxes and emotion labels
        
        Args:
            frame: Video frame
            video_results: List of video detection results
            
        Returns:
            Frame with face detection overlays
        """
        if not video_results:
            return frame
        
        for i, result in enumerate(video_results):
            if 'box' in result and 'dominant_emotion' in result:
                x, y, w, h = result['box']
                emotion = result['dominant_emotion']
                confidence = result.get('confidence', 1.0)
                
                # Get color for emotion
                color = self.emotion_colors.get(emotion, self.default_color)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw emotion label
                label = f"{emotion} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _draw_fused_emotion(self, frame, fused_emotion_result):
        """
        Draw the fused emotion result
        
        Args:
            frame: Video frame
            fused_emotion_result: Fused emotion result dictionary
            
        Returns:
            Frame with fused emotion overlay
        """
        if not fused_emotion_result or 'fused_emotion' not in fused_emotion_result:
            return frame
        
        emotion = fused_emotion_result['fused_emotion']
        confidence = fused_emotion_result.get('confidence', 0.5)
        source_emotions = fused_emotion_result.get('source_emotions', {})
        
        # Get color for fused emotion
        color = self.emotion_colors.get(emotion, self.default_color)
        
        # Draw main fused emotion
        main_text = f"Overall: {emotion.upper()} ({confidence:.2f})"
        cv2.putText(frame, main_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw source emotions
        y_offset = 60
        for modality, mod_emotion in source_emotions.items():
            mod_color = self.emotion_colors.get(mod_emotion, self.default_color)
            source_text = f"{modality.capitalize()}: {mod_emotion}"
            cv2.putText(frame, source_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mod_color, 2)
            y_offset += 25
        
        return frame
    
    def _draw_summary_text(self, frame, summary_text):
        """
        Draw summary text on the frame
        
        Args:
            frame: Video frame
            summary_text: Summary text to display
            
        Returns:
            Frame with summary text overlay
        """
        if not summary_text:
            return frame
        
        # Split text into lines
        lines = summary_text.split('\n')
        
        # Draw background rectangle
        line_height = 25
        text_width = max(len(line) for line in lines) * 10
        text_height = len(lines) * line_height
        
        # Position in top-right corner
        x = frame.shape[1] - text_width - 20
        y = 30
        
        # Draw background
        cv2.rectangle(frame, (x - 10, y - 20), (x + text_width + 10, y + text_height + 10), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 10, y - 20), (x + text_width + 10, y + text_height + 10), 
                     (255, 255, 255), 2)
        
        # Draw text lines
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                cv2.putText(frame, line, (x, y + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_frame_info(self, frame):
        """
        Draw basic frame information
        
        Args:
            frame: Video frame
            
        Returns:
            Frame with frame info overlay
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Draw frame info in bottom-left corner
        info_text = f"Frame: {width}x{height}"
        cv2.putText(frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def create_emotion_legend(self, frame):
        """
        Create a legend showing emotion colors
        
        Args:
            frame: Video frame
            
        Returns:
            Frame with emotion legend
        """
        # Position legend in bottom-right corner
        start_x = frame.shape[1] - 200
        start_y = frame.shape[0] - 200
        
        # Draw legend background
        cv2.rectangle(frame, (start_x - 10, start_y - 10), 
                     (start_x + 190, start_y + 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (start_x - 10, start_y - 10), 
                     (start_x + 190, start_y + 180), (255, 255, 255), 2)
        
        # Draw legend title
        cv2.putText(frame, "Emotion Colors", (start_x, start_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw emotion color swatches
        y_offset = start_y + 35
        for emotion, color in self.emotion_colors.items():
            # Draw color swatch
            cv2.rectangle(frame, (start_x, y_offset - 10), 
                         (start_x + 20, y_offset + 5), color, -1)
            cv2.rectangle(frame, (start_x, y_offset - 10), 
                         (start_x + 20, y_offset + 5), (255, 255, 255), 1)
            
            # Draw emotion name
            cv2.putText(frame, emotion.capitalize(), (start_x + 25, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
        
        return frame
