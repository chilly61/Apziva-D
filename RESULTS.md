# Apziva Project D - Results Summary

## Part 1: MonReader (Video Flip Detection)

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total Segments | 194 |
| Total Images | 2,804 |
| Training Segments | 117 |
| Testing Segments | 77 |
| Flip Segments | 90 |
| NotFlip Segments | 104 |

### Model Comparison

| Method | Feature Type | Accuracy | F1 Score |
|--------|-------------|----------|----------|
| HOG + RandomForest | HOG + Color Histogram | 97.40% | 95.83% |
| ResNet-50 + RandomForest | ResNet-50 (Avg Pool) | 94.81% | 91.30% |
| **ResNet-50 + LSTM** | ResNet-50 Sequence | **98.70%** | **97.96%** |

**Best Model**: LSTM with 98.70% accuracy

---

## Part 2: Book OCR (Text Extraction)

### OCR Methods Compared

| Method | Type | Word Accuracy | CER | Selected |
|--------|------|---------------|-----|----------|
| EasyOCR | Traditional | ~55% | ~75% | ❌ |
| PaddleOCR | Traditional | N/A | >90% | ❌ |
| **Llama-4-Scout** (Groq) | VLM | **91.66%** | **15.21%** | ✅ |

**Selected**: Llama-4-Scout via Groq API

---

## Part 3: Text-to-Speech

| Method | Status | Notes |
|--------|--------|-------|
| gTTS | ✅ Selected | Free, works well |
| Sesame CSM | ⚠️ Limited | ~10s limit |
| Dia | ❌ Issues | Compatibility problems |

**Selected**: gTTS (free, no limits)

---

## Part 4: Blind Reader API

- Deployed to HuggingFace Spaces: https://huggingface.co/spaces/Chilly61/MonReader
- Uses: Groq API (OCR) + gTTS (TTS)

---

## Project Timeline

1. **Part 1**: CNN/LSTM flip detection → 98.7% accuracy
2. **Part 2**: OCR comparison → VLM selected (Llama-4-Scout)
3. **Part 3**: TTS comparison → gTTS selected
4. **Part 4**: Complete web application deployed