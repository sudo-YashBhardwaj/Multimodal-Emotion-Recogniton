import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Initialize TEA model
        try:
            tea_model_name = config.get('tea_model', 'SamLowe/roberta-base-go_emotions')
            self.tokenizer = AutoTokenizer.from_pretrained(tea_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(tea_model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading TEA model: {e}")
            self.tokenizer = None
            self.model = None
    
    def process_text(self, text, timestamp):
        """
        Process text for emotion analysis
        
        Args:
            text: Input text string
            timestamp: Timestamp of the text
            
        Returns:
            Dictionary containing emotion analysis results
        """
        result = {
            'timestamp': timestamp,
            'dominant_emotion': 'neutral',
            'dominant_emotion_score': 0.0
        }
        
        if not text or not text.strip():
            return result
        
        try:
            # Tokenize and analyze text
            emotion_result = self._analyze_text_emotion(text)
            
            if emotion_result:
                result['dominant_emotion'] = emotion_result['emotion']
                result['dominant_emotion_score'] = emotion_result['score']
        
        except Exception as e:
            print(f"Error processing text: {e}")
        
        return result
    
    def _analyze_text_emotion(self, text):
        """
        Analyze emotion in text using the loaded model
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with emotion and score, or None if error
        """
        if self.model is None or self.tokenizer is None:
            return None
        
        try:
            # Tokenize input text
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get the highest scoring emotion
                max_score, max_idx = torch.max(probabilities, dim=-1)
                
                # Get emotion label
                emotion_label = self.model.config.id2label[max_idx.item()]
                emotion_score = max_score.item()
                
                return {
                    'emotion': emotion_label,
                    'score': emotion_score
                }
        
        except Exception as e:
            print(f"Error analyzing text emotion: {e}")
            return None