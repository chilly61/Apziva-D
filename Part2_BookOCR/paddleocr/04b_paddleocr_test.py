from paddleocr import PaddleOCR
import os
from PIL import Image
import time
import json
import cv2
import numpy as np
import re

# ==================== Configuration ====================
# Directory paths
IMAGE_DIR = 'C:\\Users\\75346\\Desktop\\Apziva Project D\\images'
LABEL_DIR = 'C:\\Users\\75346\\Desktop\\Apziva Project D\\labels'
OUTPUT_DIR = 'C:\\Users\\75346\\Desktop\\Apziva Project D\\paddleocr_results'

# Number of images to process (set to None for all)
NUM_IMAGES = 10  # Change to e.g., 10 or None for all 156 images

# PaddleOCR Parameters:
# --------------------
# use_textline_orientation: Whether to detect text line orientation
#   - True: Automatically detect and correct rotated text
#   - False: Assume horizontal text
USE_TEXTLINE_ORIENTATION = True

# lang: Language for OCR
#   - 'en': English
#   - 'ch': Chinese
#   - 'japan': Japanese
#   - 'korean': Korean
LANG = 'en'

# rec_batch_num: Batch size for recognition
#   - Higher = faster but more memory
REC_BATCH_NUM = 6

# det_db_thresh: Detection confidence threshold
#   - Range: 0-1, higher = more strict
#   - 0.3: Detect more text (higher recall)
#   - 0.6: Detect only confident text (higher precision)
DET_DB_THRESH = 0.3

# det_db_box_thresh: Bounding box threshold
#   - Range: 0-1
#   - Higher = filter out smaller text regions
DET_DB_BOX_THRESH = 0.5

# det_db_unclip_ratio: Text region expansion ratio
#   - Higher = merge nearby text regions
#   - Typical range: 1.5-2.5
DET_DB_UNCLIP_RATIO = 1.5

# rec_image_shape: Input image shape for recognition
#   - Format: "h,w" (height,width)
#   - Default: "3, 64, 256" for English
REC_IMAGE_SHAPE = "3, 64, 256"

# use_angle_cls: Whether to use angle classification
#   - True: Classify text orientation
#   - False: Skip angle classification
USE_ANGLE_CLS = False

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
    
    return text


# ==================== Main Script ====================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PaddleOCR Configuration:")
print(f"  Language: {LANG}")
print(f"  Text orientation: {USE_TEXTLINE_ORIENTATION}")
print(f"  Angle classification: {USE_ANGLE_CLS}")
print(f"  Detection threshold: {DET_DB_THRESH}")
print(f"  Recognition batch: {REC_BATCH_NUM}")
print("=" * 60)

print("\nPreprocessing Configuration:")
print(f"  Enabled: {ENABLE_PREPROCESSING}")
print(f"  Deskew: {ENABLE_DESKEW}")

print("\nPostprocessing Configuration:")
print(f"  Enabled: {ENABLE_POSTPROCESSING}")
print(f"  Remove extra spaces: {REMOVE_EXTRA_SPACES}")
print(f"  Remove duplicate chars: {REMOVE_DUPLICATE_CHARS}")
print("=" * 60)

# Initialize PaddleOCR with parameters
print("\nLoading PaddleOCR model...")
ocr = PaddleOCR(
    use_textline_orientation=USE_TEXTLINE_ORIENTATION,
    lang=LANG,
    rec_batch_num=REC_BATCH_NUM,
    det_db_thresh=DET_DB_THRESH,
    det_db_box_thresh=DET_DB_BOX_THRESH,
    det_db_unclip_ratio=DET_DB_UNCLIP_RATIO
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
        ocr_result = ocr.ocr(temp_path)
    else:
        ocr_result = ocr.ocr(img_path)
    elapsed = time.time() - start
    total_time += elapsed
    
    # Extract text from OCR result
    # Format: [[(bbox, (text, confidence)), ...], ...]
    if ocr_result and ocr_result[0]:
        lines = ocr_result[0]
        raw_text = ' '.join([line[1][0] for line in lines])
        confidences = [line[1][1] for line in lines]
        num_words = len(lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    else:
        raw_text = ""
        num_words = 0
        avg_confidence = 0
    
    # Postprocess the extracted text
    extracted_text = postprocess_text(
        raw_text,
        enable_postprocessing=ENABLE_POSTPROCESSING,
        remove_extra_spaces=REMOVE_EXTRA_SPACES,
        remove_duplicate_chars=REMOVE_DUPLICATE_CHARS,
        corrections=COMMON_OCR_CORRECTIONS
    )
    
    result_data = {
        'file': img_file,
        'ground_truth': ground_truth,
        'extracted_text': extracted_text,
        'raw_text': raw_text,  # Keep raw for comparison
        'num_words_detected': num_words,
        'avg_confidence': round(avg_confidence, 4),
        'processing_time_seconds': round(elapsed, 2)
    }
    results.append(result_data)
    
    # Print progress
    print(f"[{i+1}/{len(image_files)}] {img_file}")
    print(f"    Time: {elapsed:.2f}s | Words: {num_words} | Avg Conf: {avg_confidence:.2%}")

# Calculate statistics
avg_time = total_time / len(image_files) if image_files else 0
total_words = sum(r['num_words_detected'] for r in results)
avg_confidence = sum(r['avg_confidence'] for r in results) / len(results) if results else 0

print("-" * 60)
print("\n" + "=" * 60)
print("Summary Statistics:")
print(f"  Total images processed: {len(image_files)}")
print(f"  Total processing time: {total_time:.2f}s")
print(f"  Average time per image: {avg_time:.2f}s")
print(f"  Total words detected: {total_words}")
print(f"  Average words per image: {total_words/len(image_files):.1f}" if image_files else "N/A")
print(f"  Average confidence: {avg_confidence:.2%}")
print("=" * 60)

# Save results to JSON
output_file = os.path.join(OUTPUT_DIR, 'paddleocr_results.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {output_file}")

# Print sample results
print("\n" + "=" * 60)
print("Sample Results (first 3 images):")
print("=" * 60)

for i, r in enumerate(results[:3]):
    print(f"\n[{i+1}] {r['file']}")
    print(f"    Ground Truth: {r['ground_truth'][:100]}...")
    print(f"    Extracted:   {r['extracted_text'][:100]}...")
    print(f"    Confidence:  {r['avg_confidence']:.2%}")
