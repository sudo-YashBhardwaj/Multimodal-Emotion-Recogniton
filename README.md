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
│   └── input_manager.py          # Video/audio input handling with threading
├── processing/
│   ├── video_processor.py        # MTCNN + DeepFace facial expression analysis
│   ├── audio_processor.py        # Whisper STT + RoBERTa SER
│   └── text_processor.py         # RoBERTa text emotion analysis
├── fusion/
│   └── fusion_engine.py          # Priority-based late fusion algorithm
├── analysis/
│   ├── emotion_tracker.py        # Emotion history tracking and statistics
│   └── summarizer.py             # Summary generation with sentiment analysis
└── output/
    └── display.py                # OpenCV-based real-time visualization
```

### Implementation Details

#### Input Manager (`input_manager.py`)
- **Threading**: Separate worker threads for video and audio capture
- **Queue System**: `queue.Queue(maxsize=10)` for thread-safe data transfer
- **Live Mode**: Direct webcam/microphone capture via OpenCV and SoundDevice
- **Offline Mode**: MoviePy-based audio extraction from video files
- **Synchronization**: Timestamp-based alignment of video frames and audio chunks

#### Video Processor (`video_processor.py`)
- **Face Detection**: MTCNN with confidence threshold filtering
- **Fallback Mechanism**: OpenCV Haar Cascades if MTCNN initialization fails
- **Emotion Analysis**: DeepFace ensemble model (VGG-Face, Facenet, OpenFace, DeepFace)
- **Multi-face Support**: Processes all detected faces in parallel
- **Output Format**: List of dictionaries with bounding boxes, emotions, and confidence scores

#### Audio Processor (`audio_processor.py`)
- **STT Pipeline**: Whisper model (configurable: tiny/base/small/medium/large)
- **SER Pipeline**: RoBERTa-base-go-emotions on transcribed text
- **Audio Normalization**: Automatic preprocessing (padding, normalization)
- **Device Support**: CPU/CUDA device selection via PyTorch
- **Chunk Processing**: 1024-sample blocks (~64ms at 16kHz)

#### Text Processor (`text_processor.py`)
- **Model**: RoBERTa-base-go-emotions (same as SER, different input)
- **Tokenization**: Hugging Face AutoTokenizer
- **Emotion Mapping**: Maps 28 emotion labels to 7 primary emotions
- **Confidence Scoring**: Softmax-based probability distribution

#### Fusion Engine (`fusion_engine.py`)
- **Algorithm**: Late fusion with priority-based selection
- **Priority Order**: Text (3) > Video (2) > Audio (1)
- **Neutral Filtering**: Prioritizes non-neutral emotions across modalities
- **Confidence Calculation**: Agreement-based scoring (0.5-0.9 range)
- **Fallback**: Mode-based selection if priority fails

#### Emotion Tracker (`emotion_tracker.py`)
- **Data Structure**: Time-series list of emotion events
- **Statistics**: Distribution, frequency, duration calculations
- **Timeline Analysis**: Period identification and pattern detection

#### Display Manager (`display.py`)
- **Rendering**: OpenCV drawing functions for real-time overlay
- **Visualization**: Bounding boxes, emotion labels, confidence scores
- **Interactive**: Legend toggle, summary display

## Architecture

### Processing Pipeline

```
Input → Video/Audio Capture → Multi-modal Processing → Fusion → Display
         (Threading)           (Parallel Channels)      (Late)   (OpenCV)
```

1. **Input Layer**: Separate threads for video (30fps) and audio (16kHz, 1024 sample blocks)
2. **Processing Layer**: Three parallel pipelines:
   - **Video**: MTCNN face detection → DeepFace emotion analysis
   - **Audio**: Whisper STT → RoBERTa-based SER (Speech Emotion Recognition)
   - **Text**: RoBERTa-based TEA (Text Emotion Analysis) on transcribed speech
3. **Fusion Layer**: Priority-based late fusion (Text > Video > Audio)
4. **Output Layer**: Real-time visualization with emotion overlays

### Concurrency Model

- **Multi-threaded input capture**: Separate threads for video and audio streams
- **Queue-based buffering**: Thread-safe queues (maxsize=10) prevent blocking
- **Frame dropping**: Queue full → skip frame to maintain real-time performance
- **Synchronization**: Timestamp-based alignment across modalities

### Models

| Component | Model | Architecture | Parameters | Input |
|-----------|-------|--------------|------------|-------|
| **Face Detection** | MTCNN | Multi-task CNN | ~3M | RGB frames (any resolution) |
| **Face Emotion** | DeepFace | Ensemble (VGG-Face, Facenet, OpenFace, DeepFace) | ~100M | Detected face regions |
| **Speech-to-Text** | OpenAI Whisper | Transformer (base.en) | ~74M | 16kHz mono audio |
| **Speech Emotion** | RoBERTa-base-go-emotions | RoBERTa (base) | ~125M | Transcribed text |
| **Text Emotion** | RoBERTa-base-go-emotions | RoBERTa (base) | ~125M | Text tokens |

### Fusion Algorithm

**Late Fusion Strategy**: Priority-based emotion selection

1. **Emotion Extraction**: Extract dominant emotion from each modality
2. **Neutral Filtering**: Prioritize non-neutral emotions
3. **Priority Selection**: 
   - Text modality (priority 3) - Most reliable
   - Video modality (priority 2) - Visual cues
   - Audio modality (priority 1) - Prosodic features
4. **Confidence Calculation**: 
   - Agreement-based: `0.5 + (agreement_ratio * 0.4)`
   - Max confidence: 0.9 when all modalities agree

### Audio Processing

- **Sample Rate**: 16kHz (mono)
- **Block Size**: 1024 samples (~64ms chunks)
- **Preprocessing**: Automatic normalization and padding
- **STT**: Whisper processes audio segments with timestamps
- **SER**: RoBERTa analyzes transcribed text for emotion

### Video Processing

- **Frame Rate**: Variable (typically 30fps from webcam)
- **Face Detection**: MTCNN with confidence threshold (default: 0.7)
- **Fallback**: OpenCV Haar Cascades if MTCNN fails
- **Emotion Detection**: DeepFace analyzes each detected face
- **Multi-face Support**: Processes all detected faces in frame

### Performance Characteristics

- **Latency**: ~100-300ms per frame (depends on model size)
- **Throughput**: ~2-3 fps processing (with full pipeline)
- **Memory**: ~2-4GB RAM (model loading)
- **CPU Usage**: High (multi-threaded, CPU-intensive models)
- **GPU Acceleration**: Optional CUDA support for PyTorch models

### Technical Stack

- **OpenCV** (4.8+) - Video I/O, frame processing, display
- **DeepFace** (0.0.79+) - Facial expression recognition
- **OpenAI Whisper** (20231117+) - Speech-to-text conversion
- **Hugging Face Transformers** (4.30+) - RoBERTa models for emotion analysis
- **MTCNN** (0.1.1+) - Face detection
- **MoviePy** (1.0.3+) - Video/audio extraction (offline mode)
- **PyTorch** (2.0+) - Deep learning framework
- **NumPy** (1.24+) - Numerical operations
- **SoundDevice** (0.4.6+) - Real-time audio capture

## Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: ~2GB for models (first-time download)
- **Webcam**: Required for live mode
- **Microphone**: Required for live mode audio
- **CUDA**: Optional (GPU acceleration for PyTorch models)
- **OS**: Linux, macOS, Windows (with OpenCV support)

## License

MIT License - see LICENSE file for details.
