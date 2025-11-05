# Multimodal Emotion Recognition

Real-time emotion recognition system combining facial expressions, speech tone, and text sentiment analysis from webcam feeds and video files.

## Features

- **Multimodal Analysis**: Facial expressions (DeepFace), speech emotion (Whisper + Transformers), and text sentiment
- **Real-time Processing**: Live webcam analysis with interactive visualization
- **Video File Support**: Analyze pre-recorded videos with synchronized audio processing
- **Emotion Fusion**: Intelligent combination of results from all modalities
- **Summary Generation**: Automatic emotion summaries and statistics

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Multimodal-Emotion-Recogniton

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Models will be downloaded automatically on first run.

## Usage

### Live Webcam Analysis
```bash
python main.py --source 0
```

### Video File Analysis
```bash
python main.py --source path/to/video.mp4
```

### Options
```bash
python main.py --source 0 \
    --device cuda \                    # Use GPU (cpu/cuda)
    --whisper-model base.en \         # Whisper model (tiny/base/small/medium/large)
    --detection-threshold 0.7         # Face detection threshold (0.0-1.0)
```

### Controls
- `q` - Quit
- `s` - Show emotion summary
- `l` - Toggle legend

## Project Structure

```
emotion_analyzer/
├── input/
│   └── input_manager.py          # Video/audio input handling
├── processing/
│   ├── video_processor.py        # Facial expression analysis
│   ├── audio_processor.py        # Speech-to-text and emotion recognition
│   └── text_processor.py         # Text sentiment analysis
├── fusion/
│   └── fusion_engine.py          # Multi-modal fusion
├── analysis/
│   ├── emotion_tracker.py        # Emotion history tracking
│   └── summarizer.py             # Summary generation
└── output/
    └── display.py                # Visualization
```

## Technologies

- **OpenCV** - Video I/O and display
- **DeepFace** - Facial expression recognition
- **OpenAI Whisper** - Speech-to-text
- **Hugging Face Transformers** - Text and speech emotion analysis
- **MTCNN** - Face detection
- **MoviePy** - Video/audio processing

## Requirements

- Python 3.8+
- Webcam (for live mode)
- ~4GB RAM (model loading)
- CUDA GPU (optional, for faster processing)

## License

MIT License - see LICENSE file for details.
