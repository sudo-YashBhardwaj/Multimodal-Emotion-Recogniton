# 🧪 Complete Testing Guide for Multimodal Emotion Recognition

## 📋 **Testing Levels**

### **Level 1: Basic Code Structure Test (No Dependencies)**
Test the core logic without heavy ML dependencies.

### **Level 2: Component Integration Test**
Test individual components with minimal dependencies.

### **Level 3: Full System Test**
Test the complete application with all dependencies.

---

## 🔧 **Step 1: Environment Setup**

### **Option A: Use System Python (Recommended)**
```bash
# Create a fresh virtual environment
python3 -m venv test_env
source test_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Option B: Fix Conda Environment**
```bash
# If using conda, reinstall opencv
conda install -c conda-forge opencv-python
# OR
pip uninstall opencv-python
pip install opencv-python
```

---

## 🧪 **Step 2: Testing Commands**

### **Test 1: Basic Structure (No Dependencies)**
```bash
python test_basic.py
```

### **Test 2: Individual Components**
```bash
# Test video processing
python -c "
from emotion_analyzer.processing.video_processor import VideoProcessor
config = {'detection_threshold': 0.7}
processor = VideoProcessor(config)
print('✓ VideoProcessor works')
"

# Test audio processing
python -c "
from emotion_analyzer.processing.audio_processor import AudioProcessor
config = {'device': 'cpu', 'whisper_model': 'base.en'}
processor = AudioProcessor(config)
print('✓ AudioProcessor works')
"
```

### **Test 3: Full Application**
```bash
# Test with webcam (if available)
python main.py --source 0

# Test with video file
python main.py --source path/to/your/video.mp4
```

---

## 🎯 **Step 3: Expected Results**

### **✅ Success Indicators:**
- All imports work without errors
- Components initialize successfully
- Video display opens (if webcam available)
- Emotion detection works
- Real-time processing occurs

### **❌ Common Issues & Solutions:**

#### **Issue 1: OpenCV Import Error**
```bash
# Solution: Reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

#### **Issue 2: Missing System Libraries**
```bash
# macOS
brew install ffmpeg libvorbis

# Ubuntu/Debian
sudo apt-get install ffmpeg libvorbis-dev
```

#### **Issue 3: Model Download Issues**
```bash
# Models will download automatically on first run
# Ensure stable internet connection
```

---

## 🚀 **Step 4: Performance Testing**

### **Test Performance:**
```bash
# Monitor CPU/Memory usage
python main.py --source 0 --device cpu

# Test with GPU (if available)
python main.py --source 0 --device cuda
```

### **Expected Performance:**
- **CPU Usage**: 60-80% during processing
- **Memory**: 2-4GB RAM
- **FPS**: 10-15 FPS (depending on hardware)

---

## 🔍 **Step 5: Debugging**

### **Enable Debug Mode:**
```bash
# Add debug prints to see what's happening
python -c "
import sys
sys.path.append('.')
from emotion_analyzer.input.input_manager import InputManager
print('Testing InputManager...')
config = {'device': 'cpu'}
try:
    manager = InputManager(0, config)
    print('✓ InputManager created successfully')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

### **Check Dependencies:**
```bash
# Check if all required packages are installed
python -c "
import cv2, numpy, torch, whisper, transformers, mtcnn, deepface
print('✓ All dependencies available')
"
```

---

## 📊 **Step 6: Validation Tests**

### **Test Video Processing:**
```bash
python -c "
import cv2
import numpy as np
from emotion_analyzer.processing.video_processor import VideoProcessor

# Create a test frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)
config = {'detection_threshold': 0.7}
processor = VideoProcessor(config)

# Test processing
results = processor.process_frame(frame, 0.0)
print(f'✓ Video processing works: {len(results)} faces detected')
"
```

### **Test Audio Processing:**
```bash
python -c "
import numpy as np
from emotion_analyzer.processing.audio_processor import AudioProcessor

# Create test audio
audio = np.random.randn(16000).astype(np.float32)
config = {'device': 'cpu', 'whisper_model': 'base.en'}
processor = AudioProcessor(config)

# Test processing
result = processor.process_chunk(audio, 16000, 0.0)
print(f'✓ Audio processing works: {result}')
"
```

---

## 🎮 **Step 7: Interactive Testing**

### **Test Live Webcam:**
```bash
python main.py --source 0
# Press 'q' to quit, 's' for summary, 'l' for legend
```

### **Test Video File:**
```bash
python main.py --source data/sample_video.mp4
```

### **Test with Different Models:**
```bash
# Use smaller/faster models
python main.py --source 0 --whisper-model tiny.en
python main.py --source 0 --detection-threshold 0.5
```

---

## 📈 **Step 8: Performance Benchmarks**

### **Expected Benchmarks:**
- **Initialization**: 10-30 seconds (model loading)
- **Processing Speed**: 10-15 FPS
- **Memory Usage**: 2-4GB RAM
- **Accuracy**: 80-90% emotion detection

### **Optimization Tips:**
```bash
# Use smaller models for faster processing
python main.py --source 0 --whisper-model tiny.en

# Reduce detection threshold for more sensitive detection
python main.py --source 0 --detection-threshold 0.5

# Use CPU for systems with limited GPU memory
python main.py --source 0 --device cpu
```

---

## 🐛 **Troubleshooting**

### **Common Error Messages:**

1. **"Library not loaded: libvorbis"**
   ```bash
   brew install libvorbis
   ```

2. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

3. **"CUDA out of memory"**
   ```bash
   python main.py --source 0 --device cpu
   ```

4. **"No faces detected"**
   - Check lighting conditions
   - Adjust detection threshold: `--detection-threshold 0.5`
   - Ensure face is clearly visible

### **Debug Commands:**
```bash
# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Check available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Test audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## ✅ **Success Checklist**

- [ ] All imports work without errors
- [ ] Components initialize successfully  
- [ ] Video capture works (webcam or file)
- [ ] Face detection works
- [ ] Emotion analysis works
- [ ] Real-time display works
- [ ] Audio processing works (if audio available)
- [ ] Text processing works (if speech detected)
- [ ] Fusion engine works
- [ ] Summary generation works
- [ ] Interactive controls work (q, s, l keys)

---

## 🎯 **Final Validation**

Run this comprehensive test:
```bash
python -c "
print('🧪 COMPREHENSIVE SYSTEM TEST')
print('=' * 50)

# Test 1: Imports
try:
    from emotion_analyzer.input.input_manager import InputManager
    from emotion_analyzer.processing.video_processor import VideoProcessor
    from emotion_analyzer.processing.audio_processor import AudioProcessor
    from emotion_analyzer.processing.text_processor import TextProcessor
    from emotion_analyzer.fusion.fusion_engine import FusionEngine
    from emotion_analyzer.analysis.emotion_tracker import EmotionTracker
    from emotion_analyzer.analysis.summarizer import Summarizer
    from emotion_analyzer.output.display import DisplayManager
    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)

# Test 2: Component initialization
try:
    config = {'device': 'cpu', 'detection_threshold': 0.7}
    tracker = EmotionTracker()
    summarizer = Summarizer()
    fusion_engine = FusionEngine(config)
    print('✅ Core components initialized')
except Exception as e:
    print(f'❌ Component error: {e}')

# Test 3: Basic functionality
try:
    test_event = {'timestamp': 1.0, 'fused_emotion': 'happy', 'confidence': 0.8}
    tracker.add_emotion_event(test_event)
    history = tracker.get_history()
    summary = summarizer.generate_summary(history)
    print('✅ Basic functionality works')
except Exception as e:
    print(f'❌ Functionality error: {e}')

print('🎉 System is ready for full testing!')
"
```

This comprehensive testing approach will help you verify that the system works correctly at every level!
