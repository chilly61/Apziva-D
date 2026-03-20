"""
VLM Testing - Book OCR Text Extraction (for blind reading assistance)
Run this on Google Colab

Dataset: MLap/Book-Scan-OCR
Task: Extract text from book pages and compare with ground truth

Evaluation Metrics:
- WER: Word Error Rate (word-level accuracy, allows reordering)
- CER: Character Error Rate (character-level accuracy)
- Paragraph Analysis: Compare paragraph structure
- Text Density Analysis: Compare text density patterns
"""

from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import torch
import os
import re
from difflib import SequenceMatcher

# ==================== MOUNT DRIVE ====================
from google.colab import drive
drive.mount('/content/drive')

# ==================== CONFIG ====================
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
IMAGE_DIR = "/content/drive/MyDrive/book_ocr_images"
OUTPUT_FILE = "/content/drive/MyDrive/book_ocr_results.json"

# ==================== LOAD MODEL ====================
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded!")

# ==================== LOAD METADATA ====================
with open(os.path.join(IMAGE_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Loaded {len(metadata)} samples")

# ==================== IMPROVED SCORING FUNCTIONS ====================

def word_accuracy_wer(prediction, ground_truth):
    """
    Word Accuracy (WER-based)
    Allows reordering - compares as sets
    Higher is better (1.0 = perfect)
    """
    pred_words = set(prediction.lower().split())
    gt_words = set(ground_truth.lower().split())
    if len(gt_words) == 0:
        return 0
    # Accuracy = correct words / total gt words
    return len(pred_words & gt_words) / len(gt_words)


def character_accuracy_cer(prediction, ground_truth):
    """
    Character Accuracy (CER-based)
    Uses sequence matching for better comparison
    Higher is better (1.0 = perfect)
    """
    matcher = SequenceMatcher(None, ground_truth.lower(), prediction.lower())
    return matcher.ratio()


def paragraph_analysis(prediction, ground_truth):
    """
    Paragraph Structure Analysis
    Compares number of paragraphs and average paragraph length
    """
    # Split by common delimiters (newlines, multiple spaces)
    pred_paragraphs = re.split(r'\n+', prediction)
    gt_paragraphs = re.split(r'\n+', ground_truth)
    
    # Filter empty paragraphs
    pred_paragraphs = [p.strip() for p in pred_paragraphs if p.strip()]
    gt_paragraphs = [p.strip() for p in gt_paragraphs if p.strip()]
    
    # Compare paragraph counts
    pred_count = len(pred_paragraphs)
    gt_count = len(gt_paragraphs)
    count_ratio = pred_count / max(gt_count, 1)
    
    # Compare average paragraph length
    pred_avg_len = sum(len(p) for p in pred_paragraphs) / max(pred_count, 1)
    gt_avg_len = sum(len(p) for p in gt_paragraphs) / max(gt_count, 1)
    
    # Length difference ratio (0 = same, 1 = very different)
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
    """
    Text Density Analysis
    Compares average characters per word (density)
    Helps detect layout issues like double columns
    """
    pred_words = prediction.lower().split()
    gt_words = ground_truth.lower().split()
    
    pred_word_count = len(pred_words)
    gt_word_count = len(gt_words)
    
    # Average characters per word
    if pred_word_count > 0:
        pred_density = len(prediction) / pred_word_count
    else:
        pred_density = 0
    
    if gt_word_count > 0:
        gt_density = len(ground_truth) / gt_word_count
    else:
        gt_density = 0
    
    # Word count ratio
    word_count_ratio = pred_word_count / max(gt_word_count, 1)
    
    # Density difference
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


def overall_score(word_acc, char_acc, layout_warning, density_warning):
    """
    Calculate overall score
    Penalize if layout issues detected
    """
    base_score = (word_acc * 0.5 + char_acc * 0.5)
    
    # If layout warnings, reduce score slightly
    warning_penalty = 0
    if layout_warning:
        warning_penalty += 0.05
    if density_warning:
        warning_penalty += 0.05
    
    return max(0, base_score - warning_penalty)


# ==================== EXTRACTION PROMPT ====================
EXTRACTION_PROMPT = """Extract ALL visible text from this book page. Return only the text content, no descriptions."""

# ==================== RUN INFERENCE ====================
results = []
total_word_acc = 0
total_char_acc = 0
total_paragraph_diff = 0
total_density_diff = 0

for item in metadata:
    image_path = os.path.join(IMAGE_DIR, item["filename"])
    image = Image.open(image_path)
    ground_truth = item["ground_truth"]
    
    # Create prompt for text extraction
    prompt = f"USER: <image>\n{EXTRACTION_PROMPT}\nASSISTANT:"
    
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500)
    
    prediction = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract only the answer part
    if "ASSISTANT:" in prediction:
        prediction = prediction.split("ASSISTANT:")[-1].strip()
    else:
        prediction = prediction.replace(prompt, "").strip()
    
    # ============ SCORING ============
    # 1. WER-based accuracy (allows reordering)
    word_acc = word_accuracy_wer(prediction, ground_truth)
    
    # 2. CER-based accuracy (character level)
    char_acc = character_accuracy_cer(prediction, ground_truth)
    
    # 3. Paragraph structure analysis
    para_analysis = paragraph_analysis(prediction, ground_truth)
    
    # 4. Text density analysis
    density_analysis = text_density_analysis(prediction, ground_truth)
    
    # 5. Overall score
    overall = overall_score(word_acc, char_acc, para_analysis["structure_warning"], density_analysis["density_warning"])
    
    # Accumulate for averaging
    total_word_acc += word_acc
    total_char_acc += char_acc
    
    results.append({
        "filename": item["filename"],
        "ground_truth": ground_truth[:300] + "..." if len(ground_truth) > 300 else ground_truth,
        "prediction": prediction[:300] + "..." if len(prediction) > 300 else prediction,
        
        # Main metrics
        "word_accuracy": round(word_acc, 4),
        "char_accuracy": round(char_acc, 4),
        "overall_score": round(overall, 4),
        
        # Layout analysis
        "paragraph_analysis": para_analysis,
        "density_analysis": density_analysis,
        
        # Warnings
        "layout_warning": para_analysis["structure_warning"],
        "density_warning": density_analysis["density_warning"]
    })
    
    # Print summary
    warning_flags = []
    if para_analysis["structure_warning"]:
        warning_flags.append("LAYOUT")
    if density_analysis["density_warning"]:
        warning_flags.append("DENSITY")
    
    warning_str = f" [{', '.join(warning_flags)}]" if warning_flags else ""
    
    print(f"✓ {item['filename']}: WER={word_acc:.2f}, CER={char_acc:.2f}, Overall={overall:.2f}{warning_str}")

# ==================== SAVE RESULTS ====================
total = len(results)
avg_word_acc = total_word_acc / total * 100
avg_char_acc = total_char_acc / total * 100

# Count warnings
layout_warnings = sum(1 for r in results if r["layout_warning"])
density_warnings = sum(1 for r in results if r["density_warning"])

output_data = {
    "model": MODEL_NAME,
    "task": "Book OCR - Text Extraction with Layout Analysis",
    "total_samples": total,
    
    # Main metrics (average)
    "average_word_accuracy": f"{avg_word_acc:.2f}%",
    "average_char_accuracy": f"{avg_char_acc:.2f}%",
    
    # Layout analysis summary
    "layout_warning_count": f"{layout_warnings}/{total}",
    "density_warning_count": f"{density_warnings}/{total}",
    
    # Detailed results
    "results": results
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"Model: {MODEL_NAME}")
print(f"Task: Book OCR - Text Extraction with Layout Analysis")
print(f"Total samples: {total}")
print(f"\n--- Main Metrics ---")
print(f"Average Word Accuracy (WER): {avg_word_acc:.2f}%")
print(f"Average Char Accuracy (CER): {avg_char_acc:.2f}%")
print(f"\n--- Layout Analysis ---")
print(f"Layout Warning (paragraph structure): {layout_warnings}/{total}")
print(f"Density Warning (text density): {density_warnings}/{total}")
print(f"\nResults saved to: {OUTPUT_FILE}")
print(f"\nNote: Warnings indicate potential layout issues (double columns, missing paragraphs, etc.)")
