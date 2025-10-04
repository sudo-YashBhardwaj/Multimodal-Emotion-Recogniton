# Real-Time Multimodal Emotion Recognition and Summarization

A comprehensive Python application that performs real-time emotion recognition using three modalities: video (facial expressions), audio (speech tone), and text (transcribed speech content). The application can analyze both live webcam feeds and pre-recorded video files, providing real-time emotion analysis and comprehensive summaries.

## Features

- **Multimodal Analysis**: Combines facial expression recognition, speech emotion recognition, and text emotion analysis
- **Real-time Processing**: Live emotion detection and display
- **Flexible Input**: Supports both live webcam and pre-recorded video files
- **Advanced Fusion**: Intelligent combination of results from all modalities
- **Comprehensive Summaries**: Detailed emotion analysis reports with statistics and insights
- **Interactive Display**: Real-time visualization with emotion overlays and color-coded results

## Technologies Used

- **OpenCV**: Video I/O and display
- **DeepFace**: Facial expression recognition
- **MTCNN**: Face detection
- **OpenAI Whisper**: Speech-to-text conversion
- **Hugging Face Transformers**: Text and speech emotion analysis
- **MoviePy**: Video processing and audio extraction
- **NumPy**: Data manipulation
- **Threading**: Concurrent input processing

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MultimodalEmotionRecognition
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models** (first run will download automatically):
   - Whisper models for speech-to-text
   - Hugging Face models for emotion analysis
   - DeepFace models for facial expression recognition

## Usage

### Basic Usage

**Live webcam analysis**:
```bash
python main.py --source 0
```

**Pre-recorded video analysis**:
```bash
python main.py --source path/to/your/video.mp4
```

### Advanced Options

```bash
python main.py --source 0 \
    --device cuda \
    --whisper-model base.en \
    --detection-threshold 0.8
```

### Command Line Arguments

- `--source`: Video source (webcam device number or video file path)
- `--device`: Processing device (`cpu` or `cuda`)
- `--whisper-model`: Whisper model for STT (`tiny`, `base`, `small`, `medium`, `large`)
- `--detection-threshold`: Face detection confidence threshold (0.0-1.0)

### Interactive Controls

During execution, you can use these keyboard shortcuts:
- `q`: Quit the application
- `s`: Show current emotion summary
- `l`: Toggle emotion legend display

## Project Structure

```
multimodal-emotion-analyzer/
├── README.md
├── requirements.txt
├── main.py
└── emotion_analyzer/
    ├── __init__.py
    ├── input/
    │   ├── __init__.py
    │   └── input_manager.py          # Handles video/audio input
    ├── processing/
    │   ├── __init__.py
    │   ├── video_processor.py        # Facial expression analysis
    │   ├── audio_processor.py        # Speech-to-text and SER
    │   └── text_processor.py         # Text emotion analysis
    ├── fusion/
    │   ├── __init__.py
    │   └── fusion_engine.py          # Combines all modalities
    ├── analysis/
    │   ├── __init__.py
    │   ├── emotion_tracker.py        # Tracks emotion history
    │   └── summarizer.py             # Generates summaries
    └── output/
        ├── __init__.py
        └── display.py                # Visualization and display
```

## Components Overview

### Input Manager (`input_manager.py`)
- Manages both live and offline video/audio streams
- Handles threading for concurrent processing
- Extracts audio from video files for offline analysis

### Video Processor (`video_processor.py`)
- Uses MTCNN for face detection
- Employs DeepFace for facial expression recognition
- Provides fallback detection methods

### Audio Processor (`audio_processor.py`)
- Implements OpenAI Whisper for speech-to-text
- Performs speech emotion recognition
- Handles audio preprocessing and normalization

### Text Processor (`text_processor.py`)
- Analyzes transcribed text for emotional content
- Uses Hugging Face transformers for text emotion analysis
- Provides confidence scores for predictions

### Fusion Engine (`fusion_engine.py`)
- Combines results from all three modalities
- Implements priority-based fusion strategy
- Calculates confidence scores for fused results

### Emotion Tracker (`emotion_tracker.py`)
- Maintains history of emotion events
- Provides statistics and timeline analysis
- Identifies emotion periods and patterns

### Summarizer (`summarizer.py`)
- Generates comprehensive emotion summaries
- Identifies strong emotion periods
- Provides sentiment assessment

### Display Manager (`display.py`)
- Handles real-time visualization
- Draws emotion overlays and bounding boxes
- Provides interactive legend and controls

## Example Output

```
=== Emotion Analysis Summary ===
Session Duration: 120.5 seconds
Total Emotion Events: 245

=== Overall Emotion Distribution ===
- Happy: 89 events (36.3%)
- Neutral: 67 events (27.3%)
- Surprise: 34 events (13.9%)
- Sad: 28 events (11.4%)
- Angry: 27 events (11.0%)

=== Key Findings ===
- Most frequent emotion: Happy (36.3%)
- No significant strong emotion periods detected

=== Overall Sentiment Assessment ===
- Positive sentiment (positive: 36.3%, negative: 22.4%, neutral: 27.3%)
```

## Performance Considerations

- **CPU Usage**: The application is CPU-intensive due to multiple ML models
- **Memory**: Requires significant RAM for model loading and processing
- **GPU Support**: CUDA support available for faster processing
- **Real-time Performance**: May experience latency with complex models

## Troubleshooting

### Common Issues

1. **Model Download Errors**: Ensure stable internet connection for initial model downloads
2. **CUDA Out of Memory**: Use `--device cpu` for systems with limited GPU memory
3. **Audio Issues**: Check microphone permissions and audio device availability
4. **Face Detection Issues**: Adjust `--detection-threshold` parameter

### Performance Optimization

- Use smaller Whisper models (`tiny`, `base`) for faster processing
- Reduce detection threshold for more sensitive face detection
- Use CPU processing if GPU memory is limited
- Close other applications to free up system resources

## Future Enhancements

- Real-time emotion trend visualization
- Export functionality for emotion data
- Custom emotion model training
- Multi-person emotion tracking
- Integration with video conferencing platforms
- Emotion-based content recommendations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper speech recognition model
- Hugging Face for transformer models and libraries
- DeepFace team for facial expression recognition
- OpenCV community for computer vision tools
