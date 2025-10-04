import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.detection_threshold = config.get('detection_threshold', 0.7)
        
        # Initialize MTCNN detector
        try:
            self.mtcnn_detector = MTCNN()
        except Exception as e:
            print(f"Warning: Could not initialize MTCNN: {e}")
            self.mtcnn_detector = None
    
    def process_frame(self, frame, timestamp):
        """
        Process a video frame to detect faces and analyze emotions
        
        Args:
            frame: BGR numpy array representing the video frame
            timestamp: Timestamp of the frame
            
        Returns:
            List of dictionaries containing face detection and emotion analysis results
        """
        results = []
        
        try:
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Primary path: Use MTCNN for face detection
            if self.mtcnn_detector is not None:
                try:
                    # Detect faces using MTCNN
                    detections = self.mtcnn_detector.detect_faces(rgb_frame)
                    
                    if detections:
                        for detection in detections:
                            # Extract face bounding box
                            box = detection['box']
                            confidence = detection['confidence']
                            
                            if confidence >= self.detection_threshold:
                                x, y, w, h = box
                                
                                # Ensure coordinates are within frame bounds
                                x = max(0, int(x))
                                y = max(0, int(y))
                                w = min(w, frame.shape[1] - x)
                                h = min(h, frame.shape[0] - y)
                                
                                if w > 0 and h > 0:
                                    # Crop face region from original BGR frame
                                    face_crop = frame[y:y+h, x:x+w]
                                    
                                    # Analyze emotion on cropped face
                                    emotion_result = self._analyze_emotion(face_crop)
                                    
                                    if emotion_result:
                                        result = {
                                            'box': (x, y, w, h),
                                            'dominant_emotion': emotion_result['dominant_emotion'],
                                            'emotion_probabilities': emotion_result['emotion_probabilities'],
                                            'timestamp': timestamp,
                                            'confidence': confidence
                                        }
                                        results.append(result)
                
                except Exception as e:
                    print(f"Error in MTCNN detection: {e}")
                    # Fall back to DeepFace detection
                    results = self._fallback_analysis(frame, timestamp)
            
            else:
                # Fall back to DeepFace detection if MTCNN is not available
                results = self._fallback_analysis(frame, timestamp)
        
        except Exception as e:
            print(f"Error processing video frame: {e}")
            # Return empty results on error
            pass
        
        return results
    
    def _analyze_emotion(self, face_crop):
        """
        Analyze emotion on a cropped face region
        
        Args:
            face_crop: Cropped face image (BGR format)
            
        Returns:
            Dictionary with emotion analysis results or None if failed
        """
        try:
            # Use DeepFace with enforce_detection=False since we already have a face crop
            result = DeepFace.analyze(
                face_crop,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(result, list):
                result = result[0]
            
            return {
                'dominant_emotion': result['dominant_emotion'],
                'emotion_probabilities': result['emotion']
            }
        
        except Exception as e:
            print(f"Error analyzing emotion on face crop: {e}")
            return None
    
    def _fallback_analysis(self, frame, timestamp):
        """
        Fallback analysis using DeepFace on the full frame
        
        Args:
            frame: Full video frame (BGR format)
            timestamp: Timestamp of the frame
            
        Returns:
            List of dictionaries with emotion analysis results
        """
        results = []
        
        try:
            # Use DeepFace with enforce_detection=True and retinaface backend
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=True,
                detector_backend='retinaface'
            )
            
            if isinstance(result, list):
                result = result[0]
            
            # Create a result for the detected face
            if 'region' in result:
                box = result['region']
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                
                result_dict = {
                    'box': (x, y, w, h),
                    'dominant_emotion': result['dominant_emotion'],
                    'emotion_probabilities': result['emotion'],
                    'timestamp': timestamp,
                    'confidence': 1.0  # DeepFace doesn't provide confidence for retinaface
                }
                results.append(result_dict)
        
        except Exception as e:
            print(f"Error in fallback analysis: {e}")
            pass
        
        return results