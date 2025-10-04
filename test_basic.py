#!/usr/bin/env python3
"""
Basic functionality test for the multimodal emotion recognition system
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from emotion_analyzer.input.input_manager import InputManager
        print("✓ InputManager imported successfully")
    except Exception as e:
        print(f"✗ InputManager import failed: {e}")
        return False
    
    try:
        from emotion_analyzer.processing.video_processor import VideoProcessor
        print("✓ VideoProcessor imported successfully")
    except Exception as e:
        print(f"✗ VideoProcessor import failed: {e}")
        return False
    
    try:
        from emotion_analyzer.processing.audio_processor import AudioProcessor
        print("✓ AudioProcessor imported successfully")
    except Exception as e:
        print(f"✗ AudioProcessor import failed: {e}")
        return False
    
    try:
        from emotion_analyzer.processing.text_processor import TextProcessor
        print("✓ TextProcessor imported successfully")
    except Exception as e:
        print(f"✗ TextProcessor import failed: {e}")
        return False
    
    try:
        from emotion_analyzer.fusion.fusion_engine import FusionEngine
        print("✓ FusionEngine imported successfully")
    except Exception as e:
        print(f"✗ FusionEngine import failed: {e}")
        return False
    
    try:
        from emotion_analyzer.analysis.emotion_tracker import EmotionTracker
        print("✓ EmotionTracker imported successfully")
    except Exception as e:
        print(f"✗ EmotionTracker import failed: {e}")
        return False
    
    try:
        from emotion_analyzer.analysis.summarizer import Summarizer
        print("✓ Summarizer imported successfully")
    except Exception as e:
        print(f"✗ Summarizer import failed: {e}")
        return False
    
    try:
        from emotion_analyzer.output.display import DisplayManager
        print("✓ DisplayManager imported successfully")
    except Exception as e:
        print(f"✗ DisplayManager import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without heavy dependencies"""
    print("\nTesting basic functionality...")
    
    try:
        # Test configuration
        config = {
            'device': 'cpu',
            'whisper_model': 'base.en',
            'detection_threshold': 0.7,
            'ser_model': 'SamLowe/roberta-base-go_emotions',
            'tea_model': 'SamLowe/roberta-base-go_emotions'
        }
        
        # Test EmotionTracker (no external dependencies)
        from emotion_analyzer.analysis.emotion_tracker import EmotionTracker
        tracker = EmotionTracker()
        
        # Test adding emotion events
        test_event = {
            'timestamp': 1.0,
            'fused_emotion': 'happy',
            'confidence': 0.8
        }
        tracker.add_emotion_event(test_event)
        
        history = tracker.get_history()
        assert len(history) == 1
        assert history[0]['fused_emotion'] == 'happy'
        print("✓ EmotionTracker basic functionality works")
        
        # Test Summarizer
        from emotion_analyzer.analysis.summarizer import Summarizer
        summarizer = Summarizer()
        summary = summarizer.generate_summary(history)
        assert "happy" in summary.lower()
        print("✓ Summarizer basic functionality works")
        
        # Test FusionEngine
        from emotion_analyzer.fusion.fusion_engine import FusionEngine
        fusion_engine = FusionEngine(config)
        
        # Test with sample data
        video_results = [{'dominant_emotion': 'happy', 'timestamp': 1.0}]
        audio_results = {'ser_emotion': 'neutral', 'timestamp': 1.0}
        text_results = {'dominant_emotion': 'happy', 'timestamp': 1.0}
        
        fused = fusion_engine.fuse(video_results, audio_results, text_results)
        assert 'fused_emotion' in fused
        print("✓ FusionEngine basic functionality works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("MULTIMODAL EMOTION RECOGNITION - BASIC TEST")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed!")
        return False
    
    print("\n✅ All basic tests passed!")
    print("\nNote: Full functionality requires:")
    print("- OpenCV for video processing")
    print("- DeepFace and MTCNN for face detection")
    print("- Whisper for speech-to-text")
    print("- Transformers for emotion analysis")
    print("- MoviePy for video file processing")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
