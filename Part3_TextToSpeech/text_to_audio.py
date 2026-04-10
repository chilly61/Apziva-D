# =============================================================================
# Cell 1: Mount Google Drive
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# Cell 2: Configuration
# =============================================================================
# Choose TTS method:
# 1 = Sesame CSM (best quality, ~10s limit, requires GPU)
# 2 = Piper TTS (good quality, CPU OK)
# 3 = gTTS (free, basic quality, supports long text)
TTS_METHOD = 3  # Change this to choose different method (1=Sesame, 3=gTTS)

# HuggingFace Token (required for Sesame CSM)
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your HF token

# Paths
OCR_RESULTS_FILE = "/content/drive/MyDrive/book_ocr_results.json"
AUDIO_OUTPUT_DIR = "/content/drive/MyDrive/book_ocr_audio"

import os
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

print(f"TTS Method: {TTS_METHOD}")
print(f"  1 = Sesame CSM (high quality, ~10s limit)")
print(f"  3 = gTTS (free, supports long text)")
print(f"Output directory: {AUDIO_OUTPUT_DIR}")

# =============================================================================
# Cell 3: Install Dependencies
# =============================================================================
print("\n--- Installing dependencies ---")

# gTTS (for method 3)
if TTS_METHOD == 3:
    !pip install gtts -q
    print("gTTS installed!")

# Sesame CSM (for method 1)
if TTS_METHOD == 1:
    !pip install transformers>=4.52.1 -q
    print("Sesame CSM dependencies installed!")
    
    # Login to HuggingFace
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("Logged in to HuggingFace!")

# =============================================================================
# Cell 4: Import Libraries
# =============================================================================
import json
import torch
import gc  # For garbage collection

# gTTS
if TTS_METHOD == 3:
    from gtts import gTTS
    print("gTTS imported!")

# Sesame CSM
if TTS_METHOD == 1:
    from transformers import CsmForConditionalGeneration, AutoProcessor
    print("Sesame CSM imported!")

# =============================================================================
# Cell 5: Define TTS Functions
# =============================================================================
def generate_audio_gtts(text, output_path):
    """Generate audio using Google TTS (free, supports long text)"""
    print(f"    Generating with gTTS...")
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_path)
    return True

def generate_audio_sesame(text, output_path, processor, model, device):
    """Generate audio using Sesame CSM (best quality, ~10s limit)"""
    # Add [0] for conversational context
    text_input = f"[0]{text}"
    print(f"    Input text length: {len(text_input)} chars")
    
    inputs = processor(text=text_input, add_special_tokens=True).to(device)
    print(f"    Input tokens shape: {inputs.input_ids.shape}")
    
    print(f"    Generating with Sesame CSM...")
    audio = model.generate(**inputs, output_audio=True)
    
    print(f"    Audio result type: {type(audio)}")
    
    # Save
    processor.save_audio(audio, output_path)
    return True

# =============================================================================
# Cell 6: Check GPU and Load OCR Results
# =============================================================================
print(f"\n--- GPU Status ---")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"\n--- Loading OCR Results ---")
print(f"From: {OCR_RESULTS_FILE}")
with open(OCR_RESULTS_FILE, "r", encoding="utf-8") as f:
    ocr_results = json.load(f)

print(f"Loaded {len(ocr_results['results'])} samples")

# Load model ONCE before the loop (only for Sesame CSM)
if TTS_METHOD == 1:
    MODEL_NAME = "sesame/csm-1b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n--- Loading Sesame CSM model ONCE (before loop) ---")
    print(f"    Device: {device}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = CsmForConditionalGeneration.from_pretrained(MODEL_NAME, device_map=device)
    print(f"    Model loaded successfully!")
else:
    processor = None
    model = None
    device = None

# =============================================================================
# Cell 7: Run Text to Audio Conversion
# =============================================================================
print(f"\n{'='*60}")
print(f"Converting text to audio (Method: {TTS_METHOD})...")
print(f"{'='*60}\n")

successful = 0
failed = 0
skipped = 0

for i, result in enumerate(ocr_results['results']):
    filename = result['filename']
    prediction = result.get('prediction', '')
    
    # Skip if no prediction
    if not prediction or prediction is None:
        print(f"✗ {filename}: No prediction, skipping")
        skipped += 1
        continue
    
    if isinstance(prediction, str) and prediction.startswith('Error'):
        print(f"✗ {filename}: OCR failed, skipping")
        skipped += 1
        continue
    
    # Text length settings (different for each method)
    if TTS_METHOD == 1:
        # Sesame CSM has ~10s limit
        max_chars = 500  # ~10 seconds of audio
    else:
        # gTTS supports long text
        max_chars = 10000
    
    if len(prediction) > max_chars:
        text_for_tts = prediction[:max_chars]
        print(f"    Note: Text truncated from {len(prediction)} to {max_chars} chars")
    else:
        text_for_tts = prediction
    
    # Output filename (different extension for each method)
    if TTS_METHOD == 1:
        audio_filename = filename.replace('.jpg', '.wav').replace('.png', '.wav').replace('.jpeg', '.wav')
    else:
        audio_filename = filename.replace('.jpg', '.mp3').replace('.png', '.mp3').replace('.jpeg', '.mp3')
    
    audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_filename)
    
    print(f"[{i+1}/{len(ocr_results['results'])}] {filename}")
    print(f"    Text length: {len(text_for_tts)} chars")
    
    try:
        if TTS_METHOD == 1:
            # Sesame CSM
            success = generate_audio_sesame(text_for_tts, audio_path, processor, model, device)
        else:
            # gTTS
            success = generate_audio_gtts(text_for_tts, audio_path)
        
        if success:
            print(f"    ✓ Saved: {audio_filename}")
            successful += 1
    except Exception as e:
        print(f"    ✗ Failed: {str(e)[:100]}")
        failed += 1
    
    print()

# Clean up GPU memory (only for Sesame CSM)
if TTS_METHOD == 1:
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

# =============================================================================
# Cell 8: Summary
# =============================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"TTS Method: {TTS_METHOD}")
print(f"Total: {len(ocr_results['results'])}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Skipped: {skipped}")
print(f"\nAudio saved to: {AUDIO_OUTPUT_DIR}")

# List files
print(f"\n--- Generated Files ---")
for f in sorted(os.listdir(AUDIO_OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(AUDIO_OUTPUT_DIR, f))
    print(f"  - {f} ({size/1024:.1f} KB)")
