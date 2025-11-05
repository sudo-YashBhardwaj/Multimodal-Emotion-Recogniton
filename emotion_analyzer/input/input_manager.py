import cv2
import threading
import queue
import time
import os
import tempfile
import numpy as np
import soundfile as sf
import sounddevice as sd

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    print("Warning: moviepy not available. Offline video processing will be limited.")
    VideoFileClip = None


class InputManager:
    def __init__(self, source, config):
        self.source = source
        self.config = config
        self.is_live = isinstance(source, int)
        
        # Initialize queues and events
        self.video_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10) if self.is_live else None
        self._stop_event = threading.Event()
        
        # Threading objects
        self._video_thread = None
        self._audio_thread = None
        
        # Offline mode specific attributes
        self._full_audio_data = None
        self._temp_audio_file = None
        self._video_cap = None
        self._fps = None
        self._frame_size = None
        
        if not self.is_live:
            self._preprocess_offline_source()
    
    def _preprocess_offline_source(self):
        """Extract audio from video file and prepare for processing"""
        try:
            if VideoFileClip is None:
                raise ImportError("moviepy is required for offline video processing")
            
            # Extract audio using moviepy
            video_clip = VideoFileClip(self.source)
            audio_clip = video_clip.audio
            
            if audio_clip is None:
                print("Warning: No audio track found in video file")
                self._full_audio_data = None
                self._sample_rate = 16000  # Default sample rate
            else:
                # Create temporary audio file
                self._temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                audio_clip.write_audiofile(self._temp_audio_file.name, verbose=False, logger=None)
                
                # Load audio data
                self._full_audio_data, self._sample_rate = sf.read(self._temp_audio_file.name)
                
                # Clean up audio clip
                audio_clip.close()
            
            # Get video metadata
            self._video_cap = cv2.VideoCapture(self.source)
            if not self._video_cap.isOpened():
                raise ValueError(f"Could not open video file: {self.source}")
            
            self._fps = self._video_cap.get(cv2.CAP_PROP_FPS)
            self._frame_size = (
                int(self._video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            
            # Clean up video clip
            video_clip.close()
            
        except Exception as e:
            print(f"Error preprocessing offline source: {e}")
            raise
    
    def start_capture(self):
        """Start video and audio capture threads"""
        if self.is_live:
            self._video_thread = threading.Thread(target=self._video_capture_worker_live)
            self._audio_thread = threading.Thread(target=self._audio_capture_worker_live)
        else:
            self._video_thread = threading.Thread(target=self._video_capture_worker_file)
        
        self._video_thread.start()
        if self._audio_thread:
            self._audio_thread.start()
    
    def stop_capture(self):
        """Stop capture and clean up resources"""
        self._stop_event.set()
        
        if self._video_thread:
            self._video_thread.join()
        if self._audio_thread:
            self._audio_thread.join()
        
        if self._video_cap:
            self._video_cap.release()
        
        # Clean up temporary audio file
        if self._temp_audio_file and os.path.exists(self._temp_audio_file.name):
            os.unlink(self._temp_audio_file.name)
    
    def _video_capture_worker_live(self):
        """Capture video frames from webcam"""
        cap = cv2.VideoCapture(self.source)
        
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                timestamp = time.time()
                try:
                    self.video_queue.put((frame, timestamp, None), timeout=0.1)
                except queue.Full:
                    pass  # Skip frame if queue is full
            else:
                break
        
        cap.release()
    
    def _video_capture_worker_file(self):
        """Read video frames from file"""
        frame_count = 0
        
        while not self._stop_event.is_set() and self._video_cap.isOpened():
            ret, frame = self._video_cap.read()
            if ret:
                timestamp_sec = frame_count / self._fps
                frame_duration_sec = 1.0 / self._fps
                try:
                    self.video_queue.put((frame, timestamp_sec, frame_duration_sec), timeout=0.1)
                except queue.Full:
                    pass  # Skip frame if queue is full
                frame_count += 1
            else:
                break
    
    def _audio_capture_worker_live(self):
        """Capture audio from microphone"""
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
            timestamp = time.time()
            try:
                self.audio_queue.put((indata.copy(), timestamp), timeout=0.1)
            except queue.Full:
                pass  # Skip audio chunk if queue is full
        
        try:
            with sd.InputStream(callback=audio_callback, 
                              channels=1, 
                              samplerate=16000, 
                              blocksize=1024):
                while not self._stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error in audio capture: {e}")
    
    def get_video_frame(self):
        """Get next video frame from queue"""
        try:
            return self.video_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def get_audio_chunk(self):
        """Get next audio chunk from queue (live mode only)"""
        if not self.is_live:
            return None
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def get_offline_audio_data(self):
        """Get pre-loaded audio data (offline mode only)"""
        if self.is_live:
            return None
        return self._full_audio_data, self._sample_rate