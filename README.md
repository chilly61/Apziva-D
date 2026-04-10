# Apziva Project D: MonReader & Blind Reading Assistant

A comprehensive computer vision project that evolved from video flip detection to a complete blind reading assistance system.

## Project Overview

| Part | Description | Best Result |
|------|-------------|--------------|
| Part 1 | MonReader - Video Flip Detection | 98.70% (LSTM) |
| Part 2 | Book OCR - Text Extraction | 91.66% (Llama-4-Scout) |
| Part 3 | Text-to-Speech | Dia/CSM/gTTS |
| Part 4 | Blind Reader API | Deployed |

---

## Part 1: MonReader - Video Flip Detection

### Problem
Automatically identify whether a video segment contains "flipping" content (page turning) for digital library management.

### Dataset
- 194 video segments (2,804 images)
- 117 training / 77 testing segments

### Approaches & Results

| Method | Accuracy | F1 Score |
|--------|----------|----------|
| HOG + RandomForest | 97.40% | 95.83% |
| ResNet-50 + RandomForest | 94.81% | 91.30% |
| **ResNet-50 + LSTM** | **98.70%** | **97.96%** |

**Best**: LSTM captures temporal patterns in frame sequences.

### Quick Start
```bash
cd Part1_MonReader

# Data exploration
python scripts/01_monreader_eda.py

# Feature extraction
python scripts/02_monreader_preprocess.py   # HOG
python scripts/02c_sequential_preprocess.py  # LSTM sequence

# Training
python scripts/03_monreader_train.py     # HOG + RF
python scripts/03c_lstm_train.py           # LSTM
```

---

## Part 2: Book OCR - Text Extraction

### Problem
Extract text from scanned book pages to assist blind users with reading.

### Methods Compared

| Method | Type | Word Accuracy | Selected |
|--------|------|---------------|----------|
| EasyOCR | Traditional | ~55% | ❌ |
| PaddleOCR | Traditional | N/A | ❌ |
| **Llama-4-Scout** (Groq) | VLM | **91.66%** | ✅ |

**Key Finding**: Vision Language Models (VLM) significantly outperform traditional OCR on scanned documents.

### Quick Start
```bash
cd Part2_BookOCR

# Traditional OCR (no API needed)
python easyocr/04a_easyocr_test.py
python paddleocr/04b_paddleocr_test.py

# VLM-based OCR (requires Groq API key)
python vlms/book_ocr_api.py
```

### Configuration
Get free API key from: https://console.groq.com/keys

---

## Part 3: Text-to-Speech

### Methods Tested

| Method | Status | Notes |
|--------|--------|-------|
| gTTS | ✅ Selected | Free, no limits |
| Sesame CSM | ⚠️ Limited | ~10s limit |
| Dia | ❌ Issues | Compatibility problems |

**Selected**: gTTS (free Google TTS)

### Quick Start
```bash
cd Part3_TextToSpeech

# Test gTTS
python -c "from gtts import gTTS; tts = gTTS('Hello'); tts.save('output.mp3')"
```

---

## Part 4: Blind Reader API

### Features
- Upload book page image
- Extract text using Groq API (Llama-4-Scout)
- Convert text to audio
- Download audio file

### Online Demo
🎯 **Live**: https://huggingface.co/spaces/Chilly61/MonReader

### Local Setup
```bash
cd BlindReadingAPI

# Install dependencies
pip install -r requirements.txt

# Set Groq API key
export GROQ_API_KEY="your_key_here"

# Run locally
python app.py
```

### Deployment
```bash
# Push to HuggingFace Spaces
huggingface-cli space create MonReader
git push hfpaces main
```

---

## Project Structure

```
Apziva-D/
├── Part1_MonReader/           # CNN/LSTM Flip Detection
│   ├── scripts/             # Training scripts
│   └── outputs/            # Results & figures
│
├── Part2_BookOCR/           # OCR Comparison
│   ├── easyocr/            # EasyOCR results
│   ├── paddleocr/         # PaddleOCR results
│   ├── vlms/              # VLM (Groq) results
│   └── results/           # Comparison汇总
│
├── Part3_TextToSpeech/       # TTS Methods
│   ├── gtts/              # gTTS
│   ├── csm/               # CSM
│   └── samples/           # Audio samples
│
├── BlindReadingAPI/          # Web Application
│   ├── app.py
│   ├── ocr.py
│   ├── tts.py
│   └── requirements.txt
│
├── images/                  # Sample images
├── RESULTS.md              # Results summary
└── README.md
```

---

## Technologies Used

| Part | Technologies |
|------|--------------|
| Part 1 | TensorFlow, Keras, OpenCV, scikit-learn |
| Part 2 | EasyOCR, PaddleOCR, Groq API, Llama-4-Scout |
| Part 3 | gTTS, Sesame CSM |
| Part 4 | Gradio, HuggingFace Spaces |

---

## License

MIT

---

## Acknowledgments

- Apziva for the project opportunity
- Groq for free API access
- HuggingFace for 免费Spaces hosting
