#!/usr/bin/env python3
"""
Real-Time Multimodal Emotion Recognition and Summarization

This application analyzes video sources (live webcam or pre-recorded video) 
to perform real-time emotion recognition using three modalities:
- Video (facial expressions)
- Audio (speech tone) 
- Text (transcribed speech content)

Usage:
    python main.py --source 0                    # Use webcam (device 0)
    python main.py --source path/to/video.mp4    # Use video file
"""

import argparse
import cv2
import time
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emotion_analyzer.input.input_manager import InputManager
from emotion_analyzer.processing.video_processor import VideoProcessor
from emotion_analyzer.processing.audio_processor import AudioProcessor
from emotion_analyzer.processing.text_processor import TextProcessor
from emotion_analyzer.fusion.fusion_engine import FusionEngine
from emotion_analyzer.analysis.emotion_tracker import EmotionTracker
from emotion_analyzer.analysis.summarizer import Summarizer
from emotion_analyzer.output.display import DisplayManager


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Real-Time Multimodal Emotion Recognition and Summarization"
    )
    parser.add_argument(
        '--source', 
        required=True,
        help='Video source: webcam device number (e.g., 0) or path to video file'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for processing (default: cpu)'
    )
    parser.add_argument(
        '--whisper-model',
        default='base.en',
        help='Whisper model to use for speech-to-text (default: base.en)'
    )
    parser.add_argument(
        '--detection-threshold',
        type=float,
        default=0.7,
        help='Face detection confidence threshold (default: 0.7)'
    )
    
    return parser.parse_args()


def create_config(args):
    """Create configuration dictionary from arguments"""
    config = {
        'device': args.device,
        'whisper_model': args.whisper_model,
        'detection_threshold': args.detection_threshold,
        'ser_model': 'SamLowe/roberta-base-go_emotions',
        'tea_model': 'SamLowe/roberta-base-go_emotions'
    }
    return config


def determine_source_type(source):
    """Determine if source is webcam or video file"""
    try:
        # Try to parse as integer (webcam device)
        device_id = int(source)
        return device_id
    except ValueError:
        # Assume it's a file path
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        return source


def main():
    """Main application entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine source type
    try:
        source = determine_source_type(args.source)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create configuration
    config = create_config(args)
    
    print("Initializing components...")
    
    # Initialize all components
    try:
        input_manager = InputManager(source, config)
        video_processor = VideoProcessor(config)
        audio_processor = AudioProcessor(config)
        text_processor = TextProcessor(config)
        fusion_engine = FusionEngine(config)
        emotion_tracker = EmotionTracker()
        summarizer = Summarizer()
        display_manager = DisplayManager()
        
        print("Components initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        sys.exit(1)
    
    # Start capture
    print("Starting capture...")
    input_manager.start_capture()
    
    # Main processing loop
    print("Starting emotion analysis...")
    print("Press 'q' to quit, 's' to show summary, 'l' to toggle legend")
    
    show_legend = False
    last_summary_update = 0
    summary_update_interval = 10.0  # Update summary every 10 seconds
    
    try:
        while True:
            # Get video frame
            video_data = input_manager.get_video_frame()
            if video_data is None:
                time.sleep(0.01)
                continue
            
            frame, timestamp, frame_duration = video_data
            
            # Process video frame
            video_results = video_processor.process_frame(frame, timestamp)
            
            # Process audio based on mode
            audio_results = None
            text_results = None
            
            if input_manager.is_live:
                # Live mode: get audio chunk
                audio_data = input_manager.get_audio_chunk()
                if audio_data is not None:
                    audio_chunk, audio_timestamp = audio_data
                    audio_results = audio_processor.process_chunk(
                        audio_chunk, 16000, audio_timestamp
                    )
            else:
                # Offline mode: get audio data for current timestamp
                full_audio, sample_rate = input_manager.get_offline_audio_data()
                if full_audio is not None and frame_duration is not None:
                    # Extract audio segment for current frame
                    start_sample = int(timestamp * sample_rate)
                    end_sample = int((timestamp + frame_duration) * sample_rate)
                    
                    # Ensure indices are within bounds
                    start_sample = max(0, start_sample)
                    end_sample = min(len(full_audio), end_sample)
                    
                    if end_sample > start_sample:
                        audio_segment = full_audio[start_sample:end_sample]
                        
                        if len(audio_segment) > 0:
                            audio_results = audio_processor.process_chunk(
                                audio_segment, sample_rate, timestamp
                            )
            
            # Process text if we have transcription
            if audio_results and audio_results.get('transcription'):
                text_results = text_processor.process_text(
                    audio_results['transcription'], timestamp
                )
            
            # Fuse results from all modalities
            fused_result = fusion_engine.fuse(video_results, audio_results, text_results)
            
            # Add to emotion tracker
            if fused_result:
                emotion_tracker.add_emotion_event(fused_result)
            
            # Generate periodic summary
            current_time = time.time()
            if current_time - last_summary_update > summary_update_interval:
                emotion_history = emotion_tracker.get_history()
                if emotion_history:
                    summary_text = summarizer.generate_summary(emotion_history)
                    last_summary_update = current_time
                else:
                    summary_text = ""
            else:
                summary_text = ""
            
            # Draw results on frame
            display_frame = display_manager.draw_results(
                frame, video_results, fused_result, summary_text
            )
            
            # Add legend if requested
            if show_legend:
                display_frame = display_manager.create_emotion_legend(display_frame)
            
            # Display frame
            cv2.imshow('Multimodal Emotion Recognition', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Show current summary
                emotion_history = emotion_tracker.get_history()
                if emotion_history:
                    summary = summarizer.generate_summary(emotion_history)
                    print("\n" + "="*50)
                    print("CURRENT EMOTION SUMMARY")
                    print("="*50)
                    print(summary)
                    print("="*50)
                else:
                    print("No emotion data available yet.")
            elif key == ord('l'):
                # Toggle legend
                show_legend = not show_legend
                print(f"Legend {'enabled' if show_legend else 'disabled'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        print("Stopping capture...")
        input_manager.stop_capture()
        cv2.destroyAllWindows()
        
        # Generate final summary
        print("\nGenerating final summary...")
        emotion_history = emotion_tracker.get_history()
        
        if emotion_history:
            final_summary = summarizer.generate_summary(emotion_history)
            print("\n" + "="*60)
            print("FINAL EMOTION ANALYSIS SUMMARY")
            print("="*60)
            print(final_summary)
            print("="*60)
        else:
            print("No emotion data was collected during the session.")
        
        print("Analysis complete!")


if __name__ == "__main__":
    main()