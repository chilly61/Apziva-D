"""
Book OCR - Text Extraction API Version
Using HuggingFace Inference API instead of local model
For blind reading assistance

Reference: test_book_ocr.py (original)
Change: Local model -> API call (like KIMI_API.ipynb)
"""

from PIL import Image
import json
import os
import time
import base64
import re
from difflib import SequenceMatcher
from groq import Groq

# ==================== CONFIG ====================
# API Configuration
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your Groq API key
MODEL_NAME = "llama-3.2-90b-vision-preview"  # Groq vision model

# Paths
IMAGE_DIR = "/content/drive/MyDrive/book_ocr_images"
OUTPUT_FILE = "/content/drive/MyDrive/book_ocr_results.json"

# ==================== SETUP CLIENT ====================
print("Setting up Groq API client...")
client = Groq(api_key=GROQ_API_KEY)
print(f"Using model: {MODEL_NAME}")

# ==================== LOAD METADATA ====================
with open(os.path.join(IMAGE_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Loaded {len(metadata)} samples")

# ==================== API FUNCTION ====================
def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def call_vision_api(image_path, prompt, model=MODEL_NAME, max_retries=3):
    """Call Groq VL model via API"""
    
    # Read and encode image
    image_base64 = encode_image(image_path)
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                wait_time = (attempt + 1) * 30
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return f"Error: {error_msg}"
    
    return "Error: Max retries exceeded"

# ==================== SCORING FUNCTIONS ====================

def word_accuracy_wer(prediction, ground_truth):
    """Word Accuracy (WER-based)"""
    pred_words = set(prediction.lower().split())
    gt_words = set(ground_truth.lower().split())
    if len(gt_words) == 0:
        return 0
    return len(pred_words & gt_words) / len(gt_words)


def character_accuracy_cer(prediction, ground_truth):
    """Character Accuracy (CER-based)"""
    matcher = SequenceMatcher(None, ground_truth.lower(), prediction.lower())
    return matcher.ratio()


def paragraph_analysis(prediction, ground_truth):
    """Paragraph Structure Analysis"""
    pred_paragraphs = re.split(r'\n+', prediction)
    gt_paragraphs = re.split(r'\n+', ground_truth)
    
    pred_paragraphs = [p.strip() for p in pred_paragraphs if p.strip()]
    gt_paragraphs = [p.strip() for p in gt_paragraphs if p.strip()]
    
    pred_count = len(pred_paragraphs)
    gt_count = len(gt_paragraphs)
    count_ratio = pred_count / max(gt_count, 1)
    
    pred_avg_len = sum(len(p) for p in pred_paragraphs) / max(pred_count, 1)
    gt_avg_len = sum(len(p) for p in gt_paragraphs) / max(gt_count, 1)
    
    if gt_avg_len > 0:
        length_diff_ratio = abs(pred_avg_len - gt_avg_len) / gt_avg_len
    else:
        length_diff_ratio = 0
    
    return {
        "pred_paragraphs": pred_count,
        "gt_paragraphs": gt_count,
        "count_ratio": round(count_ratio, 2),
        "avg_length_diff_ratio": round(length_diff_ratio, 2),
        "structure_warning": count_ratio < 0.5 or count_ratio > 2 or length_diff_ratio > 0.5
    }


def text_density_analysis(prediction, ground_truth):
    """Text Density Analysis"""
    pred_words = prediction.lower().split()
    gt_words = ground_truth.lower().split()
    
    pred_word_count = len(pred_words)
    gt_word_count = len(gt_words)
    
    if pred_word_count > 0:
        pred_density = len(prediction) / pred_word_count
    else:
        pred_density = 0
    
    if gt_word_count > 0:
        gt_density = len(ground_truth) / gt_word_count
    else:
        gt_density = 0
    
    word_count_ratio = pred_word_count / max(gt_word_count, 1)
    density_diff = abs(pred_density - gt_density)
    
    return {
        "pred_word_count": pred_word_count,
        "gt_word_count": gt_word_count,
        "word_count_ratio": round(word_count_ratio, 2),
        "pred_density": round(pred_density, 2),
        "gt_density": round(gt_density, 2),
        "density_diff": round(density_diff, 2),
        "density_warning": density_diff > 2 or word_count_ratio < 0.5 or word_count_ratio > 2
    }


# ==================== EXTRACTION PROMPT ====================
# Changed from Q&A to text extraction
EXTRACTION_PROMPT = """Extract ALL visible text from this book page. Return only the raw text content, no descriptions, no summaries. Include all paragraphs."""

# ==================== RUN INFERENCE ====================
results = []
total_word_acc = 0
total_char_acc = 0

for item in metadata:
    image_path = os.path.join(IMAGE_DIR, item["filename"])
    ground_truth = item["ground_truth"]
    
    # Call API instead of local model
    prediction = call_vision_api(image_path, EXTRACTION_PROMPT)
    
    # Skip scoring if error
    if prediction.startswith("Error:"):
        print(f"✗ {item['filename']}: {prediction}")
        results.append({
            "filename": item["filename"],
            "error": prediction,
            "ground_truth": ground_truth,
            "prediction": None
        })
        continue
    
    # Scoring
    word_acc = word_accuracy_wer(prediction, ground_truth)
    char_acc = character_accuracy_cer(prediction, ground_truth)
    para_analysis = paragraph_analysis(prediction, ground_truth)
    density_analysis = text_density_analysis(prediction, ground_truth)
    
    # Accumulate
    total_word_acc += word_acc
    total_char_acc += char_acc
    
    # Warning flags
    warning_flags = []
    if para_analysis["structure_warning"]:
        warning_flags.append("LAYOUT")
    if density_analysis["density_warning"]:
        warning_flags.append("DENSITY")
    
    warning_str = f" [{', '.join(warning_flags)}]" if warning_flags else ""
    
    results.append({
        "filename": item["filename"],
        "ground_truth": ground_truth[:300] + "..." if len(ground_truth) > 300 else ground_truth,
        "prediction": prediction[:300] + "..." if len(prediction) > 300 else prediction,
        "word_accuracy": round(word_acc, 4),
        "char_accuracy": round(char_acc, 4),
        "paragraph_analysis": para_analysis,
        "density_analysis": density_analysis,
        "layout_warning": para_analysis["structure_warning"],
        "density_warning": density_analysis["density_warning"]
    })
    
    print(f"✓ {item['filename']}: WER={word_acc:.2f}, CER={char_acc:.2f}{warning_str}")

# ==================== SAVE RESULTS ====================
total = len(results)
successful = sum(1 for r in results if "error" not in r)
avg_word_acc = total_word_acc / successful * 100 if successful > 0 else 0
avg_char_acc = total_char_acc / successful * 100 if successful > 0 else 0

output_data = {
    "model": MODEL_NAME,
    "task": "Book OCR - Text Extraction via API",
    "total_samples": total,
    "successful": successful,
    "average_word_accuracy": f"{avg_word_acc:.2f}%",
    "average_char_accuracy": f"{avg_char_acc:.2f}%",
    "results": results
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"Model: {MODEL_NAME}")
print(f"Task: Book OCR - Text Extraction via API")
print(f"Total: {total}, Successful: {successful}")
print(f"Average Word Accuracy: {avg_word_acc:.2f}%")
print(f"Average Char Accuracy: {avg_char_acc:.2f}%")
print(f"\nResults saved to: {OUTPUT_FILE}")
