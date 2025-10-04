#!/usr/bin/env python3
"""
Simple test of the emotion recognition system with video file
"""

import cv2
import numpy as np
import time
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emotion_analyzer.processing.video_processor import VideoProcessor
from emotion_analyzer.fusion.fusion_engine import FusionEngine
from emotion_analyzer.analysis.emotion_tracker import EmotionTracker
from emotion_analyzer.analysis.summarizer import Summarizer
from emotion_analyzer.output.display import DisplayManager

def test_video_processing(video_path):
    """Test video processing without audio"""
    print(f"🎬 Testing video processing with: {video_path}")
    print("=" * 60)
    
    # Initialize components
    config = {
        'device': 'cpu',
        'detection_threshold': 0.7
    }
    
    try:
        video_processor = VideoProcessor(config)
        fusion_engine = FusionEngine(config)
        emotion_tracker = EmotionTracker()
        summarizer = Summarizer()
        display_manager = DisplayManager()
        
        print("✅ Components initialized")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video file: {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"📹 Video info: {duration:.1f}s, {fps:.1f} FPS, {frame_count} frames")
        
        frame_num = 0
        processed_frames = 0
        emotions_detected = 0
        
        print("\n🎯 Processing video frames...")
        print("Press 'q' to quit, 's' for summary, 'l' for legend")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            timestamp = frame_num / fps
            
            # Process every 5th frame to speed up testing
            if frame_num % 5 == 0:
                # Process video frame
                video_results = video_processor.process_frame(frame, timestamp)
                
                if video_results:
                    emotions_detected += len(video_results)
                    print(f"Frame {frame_num}: {len(video_results)} face(s) detected")
                    
                    # Create mock audio and text results for fusion
                    audio_results = {'ser_emotion': 'neutral', 'timestamp': timestamp}
                    text_results = {'dominant_emotion': 'neutral', 'timestamp': timestamp}
                    
                    # Fuse results
                    fused_result = fusion_engine.fuse(video_results, audio_results, text_results)
                    
                    # Add to tracker
                    if fused_result:
                        emotion_tracker.add_emotion_event(fused_result)
                
                processed_frames += 1
                
                # Draw results on frame
                display_frame = display_manager.draw_results(
                    frame, video_results, fused_result if 'fused_result' in locals() else None
                )
                
                # Display frame
                cv2.imshow('Emotion Recognition Test', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Show current summary
                    history = emotion_tracker.get_history()
                    if history:
                        summary = summarizer.generate_summary(history)
                        print("\n" + "="*50)
                        print("CURRENT EMOTION SUMMARY")
                        print("="*50)
                        print(summary)
                        print("="*50)
                    else:
                        print("No emotion data available yet.")
                elif key == ord('l'):
                    # Toggle legend
                    display_frame = display_manager.create_emotion_legend(display_frame)
                    cv2.imshow('Emotion Recognition Test', display_frame)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate final summary
        print("\n" + "="*60)
        print("FINAL EMOTION ANALYSIS SUMMARY")
        print("="*60)
        
        history = emotion_tracker.get_history()
        if history:
            final_summary = summarizer.generate_summary(history)
            print(final_summary)
        else:
            print("No emotion data was collected during the session.")
        
        print(f"\n📊 Processing Statistics:")
        print(f"   - Frames processed: {processed_frames}")
        print(f"   - Emotions detected: {emotions_detected}")
        print(f"   - Processing rate: {processed_frames/frame_count*100:.1f}% of frames")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    video_path = "data/sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    print("🚀 MULTIMODAL EMOTION RECOGNITION - VIDEO TEST")
    print("=" * 60)
    
    success = test_video_processing(video_path)
    
    if success:
        print("\n🎉 Video processing test completed successfully!")
    else:
        print("\n❌ Video processing test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
