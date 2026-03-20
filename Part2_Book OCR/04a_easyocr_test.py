import easyocr
import os
from PIL import Image
import time
import json
import cv2
import numpy as np
import re
from difflib import SequenceMatcher

# ==================== Configuration ====================
# Directory paths
IMAGE_DIR = 'C:\\Users\\75346\\Desktop\\Apziva Project D\\images'
LABEL_DIR = 'C:\\Users\\75346\\Desktop\\Apziva Project D\\labels'
OUTPUT_DIR = 'C:\\Users\\75346\\Desktop\\Apziva Project D\\easyocr_results'

# Number of images to process (set to None for all)
NUM_IMAGES = 10  # Change to e.g., 10 or None for all 156 images

# EasyOCR Parameters:
# ------------------
# lang_list: List of language codes to use
#   - 'en': English
#   - 'ch_sim': Simplified Chinese
#   - 'ch_tra': Traditional Chinese
#   - 'ja': Japanese, 'ko': Korean, etc.
LANG_LIST = ['en']

# gpu: Whether to use GPU for acceleration
#   - True: Use GPU (faster if NVIDIA GPU available)
#   - False: Use CPU
GPU = True

# model_storage_directory: Where to save downloaded models
#   - None: Use default location (~/.EasyOCR)
MODEL_STORAGE_DIR = None

# download_enabled: Whether to download models if not present
#   - True: Auto-download required models
#   - False: Use only existing models
DOWNLOAD_ENABLED = True

# detector: Whether to include text detection model
#   - True: Detect text regions in image
#   - False: Skip detection (faster but only works with pre-cropped images)
DETECTOR = True

# recognizer: Whether to include text recognition model
#   - True: Recognize text from detected regions
#   - False: Skip recognition
RECOGNIZER = True

# verbose: Whether to print progress messages
#   - True: Show detailed progress
#   - False: Minimal output
VERBOSE = True

# quantize: Whether to use quantized models (smaller, faster)
#   - True: Use quantized version (recommended for CPU)
#   - False: Use full precision models
QUANTIZE = False # Default = True

# batch_size: Number of images to process in a batch
#   - Higher = faster but more memory
BATCH_SIZE = 1

# ==================== Preprocessing Configuration ====================
# enable_preprocessing: Whether to apply image preprocessing
#   - True: Apply grayscale, denoising, binarization
#   - False: Use original image
ENABLE_PREPROCESSING = False

# enable_deskew: Whether to correct image skew
#   - True: Detect and correct rotated text
#   - False: Use original orientation
ENABLE_DESKEW = False

# binarization_threshold: Threshold for binary conversion
#   - 0-255, lower = more text detected but more noise
BINARIZATION_THRESHOLD = 127

# ==================== Postprocessing Configuration ====================
# enable_postprocessing: Whether to apply text postprocessing
#   - True: Apply corrections to recognized text
#   - False: Use raw OCR output
ENABLE_POSTPROCESSING = False

# remove_extra_spaces: Remove multiple consecutive spaces
#   - True: Replace multiple spaces with single space
REMOVE_EXTRA_SPACES = False

# remove_duplicate_chars: Remove repeated characters (3+ -> 2)
#   - True: 'aaaa' -> 'aa'
REMOVE_DUPLICATE_CHARS = False

# common_ocr_corrections: Dictionary of common OCR error corrections
#   - Key: incorrect text, Value: correct text
COMMON_OCR_CORRECTIONS = {
    '0': 'O', '1': 'l', '5': 'S', '8': 'B',
    'rn': 'm', 'vv': 'w', '|': 'l', '¦': 'l',
    '—': '-', '–': '-', '"': '"', '"': '"',
    ''': "'", ''': "'", '``': '"', "''": '"'
}

# ==================== Preprocessing Functions ====================

def preprocess_image(img_path, enable_preprocessing=True, enable_deskew=True, 
                     binarization_threshold=127):
    """
    Apply preprocessing to improve OCR accuracy.
    
    Args:
        img_path: Path to input image
        enable_preprocessing: Whether to apply grayscale, denoising, binarization
        enable_deskew: Whether to correct image skew
        binarization_threshold: Threshold for binarization (0-255)
    
    Returns:
        Preprocessed image as PIL Image
    """
    # Read image with OpenCV
    img = cv2.imread(img_path)
    if img is None:
        return Image.open(img_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if enable_preprocessing:
        # Apply median blur for denoising (kernel size 3x3)
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply Gaussian blur before binarization
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
        
        # Apply adaptive thresholding for binarization
        # This adapts to local lighting conditions
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Optional: Simple thresholding as fallback
        # binary = cv2.threshold(gray, binarization_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Morphological operations to clean up
        # Close small holes in text
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    else:
        binary = gray
    
    if enable_deskew:
        # Deskew: Detect text orientation and correct
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            # Adjust angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Only correct if angle is significant (> 0.5 degrees)
            if abs(angle) > 0.5:
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(
                    binary, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
    
    # Convert back to PIL Image
    return Image.fromarray(binary)


def postprocess_text(text, enable_postprocessing=True, 
                     remove_extra_spaces=True, 
                     remove_duplicate_chars=True,
                     corrections=None):
    """
    Apply postprocessing to clean up OCR output.
    
    Args:
        text: Raw text from OCR
        enable_postprocessing: Whether to apply any corrections
        remove_extra_spaces: Remove multiple consecutive spaces
        remove_duplicate_chars: Remove repeated characters (3+ -> 2)
        corrections: Dictionary of {wrong: correct} replacements
    
    Returns:
        Cleaned text string
    """
    if not enable_postprocessing:
        return text
    
    # Remove extra whitespace
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    # Remove duplicate characters (3+ -> 2)
    if remove_duplicate_chars:
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Apply common OCR corrections
    if corrections:
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
    
    # Convert to uppercase (optional, depends on use case)
    # text = text.upper()
    
    return text

# ==================== Evaluation Functions ====================

def calculate_edit_distance(text1, text2):
    """
    Calculate Levenshtein (edit) distance between two texts.
    
    The edit distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one
    word into the other.
    
    Args:
        text1: First text string
        text2: Second text string
    
    Returns:
        Integer representing the minimum edits needed
    """
    return SequenceMatcher(None, text1, text2).ratio()


def calculate_cer(ground_truth, ocr_result):
    """
    Calculate Character Error Rate (CER).
    
    CER measures the minimum number of character edits needed
    to transform the OCR result into the ground truth.
    
    Formula: CER = (Substitutions + Deletions + Insertions) / Total Characters
    
    Args:
        ground_truth: Correct text (reference)
        ocr_result: OCR output text
    
    Returns:
        Float between 0 and 1, where 0 = perfect match
    """
    if not ground_truth:
        return 1.0 if ocr_result else 0.0
    
    # Use SequenceMatcher to calculate edit distance
    matcher = SequenceMatcher(None, ground_truth, ocr_result)
    
    # Get the number of matching characters
    matches = matcher.ratio() * len(ground_truth)
    
    # CER = 1 - accuracy
    cer = 1 - (matches / len(ground_truth))
    return cer


def calculate_wer(ground_truth, ocr_result):
    """
    Calculate Word Error Rate (WER).
    
    WER is the minimum number of word edits needed to transform
    the OCR result into the ground truth.
    
    Formula: WER = (Substitutions + Deletions + Insertions) / Total Words
    
    Args:
        ground_truth: Correct text (reference)
        ocr_result: OCR output text
    
    Returns:
        Float between 0 and 1, where 0 = perfect match
    """
    gt_words = ground_truth.split()
    ocr_words = ocr_result.split()
    
    if not gt_words:
        return 1.0 if ocr_words else 0.0
    
    # Use SequenceMatcher on word lists
    matcher = SequenceMatcher(None, gt_words, ocr_words)
    
    # WER = 1 - accuracy
    wer = 1 - matcher.ratio()
    return wer


def calculate_accuracy_metrics(ground_truth, ocr_result):
    """
    Calculate comprehensive accuracy metrics for OCR results.
    
    Args:
        ground_truth: Correct text from dataset
        ocr_result: Text extracted by OCR
    
    Returns:
        Dictionary containing various accuracy metrics
    """
    # Exact match
    exact_match = 1.0 if ground_truth.strip() == ocr_result.strip() else 0.0
    
    # Character-level accuracy
    char_matching = sum(1 for a, b in zip(ground_truth, ocr_result) if a == b)
    char_accuracy = char_matching / max(len(ground_truth), len(ocr_result)) if max(len(ground_truth), len(ocr_result)) > 0 else 0
    
    # Word-level accuracy
    gt_words = set(ground_truth.lower().split())
    ocr_words = set(ocr_result.lower().split())
    word_accuracy = len(gt_words & ocr_words) / len(gt_words) if gt_words else 0
    
    # Sequence matching ratio (similar to F1)
    sequence_ratio = SequenceMatcher(None, ground_truth, ocr_result).ratio()
    
    # CER and WER
    cer = calculate_cer(ground_truth, ocr_result)
    wer = calculate_wer(ground_truth, ocr_result)
    
    return {
        'exact_match': exact_match,
        'character_accuracy': round(char_accuracy, 4),
        'word_accuracy': round(word_accuracy, 4),
        'sequence_ratio': round(sequence_ratio, 4),
        'cer': round(cer, 4),  # Character Error Rate (lower is better)
        'wer': round(wer, 4),  # Word Error Rate (lower is better)
        'gt_length': len(ground_truth),
        'ocr_length': len(ocr_result),
        'gt_word_count': len(ground_truth.split()),
        'ocr_word_count': len(ocr_result.split())
    }


# ==================== Main Script ====================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("EasyOCR Configuration:")
print(f"  Languages: {LANG_LIST}")
print(f"  GPU enabled: {GPU}")
print(f"  Detector: {DETECTOR}")
print(f"  Recognizer: {RECOGNIZER}")
print(f"  Quantize: {QUANTIZE}")
print("=" * 60)

print("\nPreprocessing Configuration:")
print(f"  Enabled: {ENABLE_PREPROCESSING}")
print(f"  Deskew: {ENABLE_DESKEW}")

print("\nPostprocessing Configuration:")
print(f"  Enabled: {ENABLE_POSTPROCESSING}")
print(f"  Remove extra spaces: {REMOVE_EXTRA_SPACES}")
print(f"  Remove duplicate chars: {REMOVE_DUPLICATE_CHARS}")
print("=" * 60)

# Initialize EasyOCR reader with parameters
print("\nLoading EasyOCR model...")
reader = easyocr.Reader(
    lang_list=LANG_LIST,
    gpu=GPU,
    model_storage_directory=MODEL_STORAGE_DIR,
    download_enabled=DOWNLOAD_ENABLED,
    detector=DETECTOR,
    recognizer=RECOGNIZER,
    verbose=VERBOSE,
    quantize=QUANTIZE
)
print("Model loaded successfully!\n")

# Get list of image files
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])

# Limit number of images if specified
if NUM_IMAGES:
    image_files = image_files[:NUM_IMAGES]

print(f"Processing {len(image_files)} images...")
print("-" * 60)

results = []
total_time = 0

for i, img_file in enumerate(image_files):
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_file = img_file + '.txt'
    label_path = os.path.join(LABEL_DIR, label_file)
    
    # Load ground truth text
    with open(label_path, 'r', encoding='utf-8') as f:
        ground_truth = f.read().strip()
    
    # Preprocess image
    if ENABLE_PREPROCESSING:
        processed_img = preprocess_image(
            img_path, 
            enable_preprocessing=True,
            enable_deskew=ENABLE_DESKEW,
            binarization_threshold=BINARIZATION_THRESHOLD
        )
        # Save preprocessed image for reference
        preprocessed_path = os.path.join(OUTPUT_DIR, f'preprocessed_{img_file}')
        processed_img.save(preprocessed_path)
    
    # Perform OCR (on preprocessed image if enabled)
    start = time.time()
    if ENABLE_PREPROCESSING:
        # Save temp processed image and OCR it
        temp_path = os.path.join(OUTPUT_DIR, 'temp_processed.png')
        processed_img.save(temp_path)
        ocr_result = reader.readtext(temp_path)
    else:
        temp_path = os.path.join(OUTPUT_DIR, 'temp_processed.png')
        ocr_result = reader.readtext(img_path)
    elapsed = time.time() - start
    total_time += elapsed
    
    # Extract text from OCR result
    # Format: [(bbox, text, confidence), ...]
    raw_text = ' '.join([item[1] for item in ocr_result])
    
    # Postprocess the extracted text
    extracted_text = postprocess_text(
        raw_text,
        enable_postprocessing=ENABLE_POSTPROCESSING,
        remove_extra_spaces=REMOVE_EXTRA_SPACES,
        remove_duplicate_chars=REMOVE_DUPLICATE_CHARS,
        corrections=COMMON_OCR_CORRECTIONS
    )
    
    confidences = [item[2] for item in ocr_result]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Calculate accuracy metrics by comparing with ground truth
    try:
        accuracy_metrics = calculate_accuracy_metrics(ground_truth, extracted_text)
    except Exception as e:
        # Fallback if calculation fails
        accuracy_metrics = {
            'exact_match': 0,
            'character_accuracy': 0,
            'word_accuracy': 0,
            'sequence_ratio': 0,
            'cer': 1.0,
            'wer': 1.0,
            'gt_length': len(ground_truth),
            'ocr_length': len(extracted_text),
            'gt_word_count': len(ground_truth.split()),
            'ocr_word_count': len(extracted_text.split())
        }
    
    result_data = {
        'file': img_file,
        'ground_truth': ground_truth,
        'extracted_text': extracted_text,
        'raw_text': raw_text,  # Keep raw for comparison
        'num_words_detected': len(ocr_result),
        'avg_confidence': round(avg_confidence, 4),
        'processing_time_seconds': round(elapsed, 2),
        # Accuracy metrics
        'exact_match': accuracy_metrics['exact_match'],
        'character_accuracy': accuracy_metrics['character_accuracy'],
        'word_accuracy': accuracy_metrics['word_accuracy'],
        'sequence_ratio': accuracy_metrics['sequence_ratio'],
        'cer': accuracy_metrics['cer'],
        'wer': accuracy_metrics['wer'],
        'gt_length': accuracy_metrics['gt_length'],
        'ocr_length': accuracy_metrics['ocr_length'],
        'gt_word_count': accuracy_metrics['gt_word_count'],
        'ocr_word_count': accuracy_metrics['ocr_word_count']
    }
    results.append(result_data)
    
    # Print progress
    print(f"[{i+1}/{len(image_files)}] {img_file}")
    print(f"    Time: {elapsed:.2f}s | Words: {len(ocr_result)} | Avg Conf: {avg_confidence:.2%}")

# Calculate statistics
avg_time = total_time / len(image_files)
total_words = sum(r['num_words_detected'] for r in results)
avg_confidence = sum(r['avg_confidence'] for r in results) / len(results)

# Calculate accuracy statistics
avg_char_accuracy = sum(r['character_accuracy'] for r in results) / len(results)
avg_word_accuracy = sum(r['word_accuracy'] for r in results) / len(results)
avg_sequence_ratio = sum(r['sequence_ratio'] for r in results) / len(results)
avg_cer = sum(r['cer'] for r in results) / len(results)
avg_wer = sum(r['wer'] for r in results) / len(results)
exact_matches = sum(r['exact_match'] for r in results)

print("-" * 60)
print("\n" + "=" * 60)
print("Summary Statistics:")
print(f"  Total images processed: {len(image_files)}")
print(f"  Total processing time: {total_time:.2f}s")
print(f"  Average time per image: {avg_time:.2f}s")
print(f"  Total words detected: {total_words}")
print(f"  Average words per image: {total_words/len(image_files):.1f}")
print(f"  Average confidence: {avg_confidence:.2%}")
print("=" * 60)
print("\n" + "=" * 60)
print("Accuracy Metrics (compared to ground truth):")
print(f"  Exact Match Rate:      {exact_matches}/{len(results)} ({exact_matches/len(results):.2%})")
print(f"  Character Accuracy:    {avg_char_accuracy:.2%}")
print(f"  Word Accuracy:         {avg_word_accuracy:.2%}")
print(f"  Sequence Ratio:        {avg_sequence_ratio:.2%}")
print(f"  Character Error Rate:  {avg_cer:.2%} (lower is better)")
print(f"  Word Error Rate:       {avg_wer:.2%} (lower is better)")
print("=" * 60)

# Save results to JSON
output_file = os.path.join(OUTPUT_DIR, 'easyocr_results.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {output_file}")

# Print sample results
print("\n" + "=" * 60)
print("Sample Results (first 3 images):")
print("=" * 60)

for i, r in enumerate(results[:3]):
    print(f"\n[{i+1}] {r['file']}")
    print(f"    Ground Truth: {r['ground_truth'][:80]}...")
    print(f"    Extracted:   {r['extracted_text'][:80]}...")
    print(f"    OCR Confidence: {r['avg_confidence']:.2%}")
    print(f"    Character Accuracy: {r['character_accuracy']:.2%}")
    print(f"    Word Accuracy:      {r['word_accuracy']:.2%}")
    print(f"    CER: {r['cer']:.2%} | WER: {r['wer']:.2%}")
