import numpy as np
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Initialize Whisper model for STT
        try:
            model_name = config.get('whisper_model', 'base.en')
            self.whisper_model = whisper.load_model(model_name, device=self.device)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.whisper_model = None
        
        # Initialize SER model
        try:
            ser_model_name = config.get('ser_model', 'SamLowe/roberta-base-go_emotions')
            self.ser_tokenizer = AutoTokenizer.from_pretrained(ser_model_name)
            self.ser_model = AutoModelForSequenceClassification.from_pretrained(ser_model_name)
            self.ser_model.to(self.device)
            self.ser_model.eval()
        except Exception as e:
            print(f"Error loading SER model: {e}")
            self.ser_tokenizer = None
            self.ser_model = None
    
    def process_chunk(self, audio_chunk, samplerate, timestamp):
        """
        Process an audio chunk for STT and SER
        
        Args:
            audio_chunk: Raw audio data (numpy array)
            samplerate: Sample rate of the audio
            timestamp: Timestamp of the audio chunk
            
        Returns:
            Dictionary containing transcription and SER results
        """
        result = {
            'timestamp': timestamp,
            'transcription': '',
            'ser_emotion': 'neutral'
        }
        
        try:
            # Preprocess audio
            audio_processed = self._preprocess_audio(audio_chunk)
            
            if audio_processed is not None:
                # Perform STT
                transcription = self._speech_to_text(audio_processed, samplerate)
                result['transcription'] = transcription
                
                # Perform SER
                ser_emotion = self._speech_emotion_recognition(audio_processed, samplerate)
                result['ser_emotion'] = ser_emotion
        
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
        
        return result
    
    def _preprocess_audio(self, audio_chunk):
        """
        Preprocess audio chunk for analysis
        
        Args:
            audio_chunk: Raw audio data
            
        Returns:
            Preprocessed audio array or None if error
        """
        try:
            # Convert to float32 and normalize to [-1.0, 1.0]
            if audio_chunk.dtype == np.int16:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            elif audio_chunk.dtype == np.int32:
                audio_float = audio_chunk.astype(np.float32) / 2147483648.0
            else:
                audio_float = audio_chunk.astype(np.float32)
            
            # Ensure audio is in the right range
            audio_float = np.clip(audio_float, -1.0, 1.0)
            
            return audio_float
        
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
    
    def _speech_to_text(self, audio_processed, samplerate):
        """
        Convert speech to text using Whisper
        
        Args:
            audio_processed: Preprocessed audio array
            samplerate: Sample rate of the audio
            
        Returns:
            Transcribed text string
        """
        if self.whisper_model is None:
            return ""
        
        try:
            # Whisper expects 16kHz audio, resample if necessary
            if samplerate != 16000:
                # Simple resampling (in production, use librosa or scipy)
                target_length = int(len(audio_processed) * 16000 / samplerate)
                audio_resampled = np.interp(
                    np.linspace(0, len(audio_processed), target_length),
                    np.arange(len(audio_processed)),
                    audio_processed
                )
            else:
                audio_resampled = audio_processed
            
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(
                audio_resampled,
                language='en',
                fp16=False
            )
            
            return result['text'].strip()
        
        except Exception as e:
            print(f"Error in speech-to-text: {e}")
            return ""
    
    def _speech_emotion_recognition(self, audio_processed, samplerate):
        """
        Perform speech emotion recognition using audio features
        
        Args:
            audio_processed: Preprocessed audio array
            samplerate: Sample rate of the audio
            
        Returns:
            Dominant emotion label
        """
        try:
            # Calculate audio features for emotion detection
            rms = np.sqrt(np.mean(audio_processed**2))
            zero_crossings = np.sum(np.diff(np.sign(audio_processed)) != 0)
            spectral_centroid = self._calculate_spectral_centroid(audio_processed, samplerate)
            
            # Enhanced heuristic-based emotion detection
            if len(audio_processed) < 100:  # Too short for reliable analysis
                return "neutral"
            
            # High energy scenarios
            if rms > 0.15:  # Very high energy
                if zero_crossings > len(audio_processed) * 0.15:  # High zero crossings
                    return "excited"
                elif spectral_centroid > 2000:  # High frequency content
                    return "angry"
                else:
                    return "surprise"
            elif rms > 0.08:  # Medium-high energy
                if zero_crossings > len(audio_processed) * 0.1:
                    return "happy"
                else:
                    return "neutral"
            elif rms < 0.02:  # Low energy
                if spectral_centroid < 1000:  # Low frequency content
                    return "sad"
                else:
                    return "neutral"
            else:  # Medium energy
                if zero_crossings < len(audio_processed) * 0.05:  # Low zero crossings
                    return "calm"
                else:
                    return "neutral"
        
        except Exception as e:
            print(f"Error in speech emotion recognition: {e}")
            return "neutral"
    
    def _calculate_spectral_centroid(self, audio, samplerate):
        """
        Calculate spectral centroid of audio signal
        
        Args:
            audio: Audio signal
            samplerate: Sample rate
            
        Returns:
            Spectral centroid value
        """
        try:
            # Simple spectral centroid calculation
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/samplerate)
            magnitude = np.abs(fft)
            
            # Calculate centroid
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
                return abs(centroid)
            else:
                return 0
        except:
            return 1000  # Default value