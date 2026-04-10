#!/usr/bin/env python3
"""Blind Reading Assistant"""

import os
from pathlib import Path
import gradio as gr

# Get API key from environment
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
if not GROQ_API_KEY:
    try:
        from config import GROQ_API_KEY
    except:
        GROQ_API_KEY = ''

TTS_LANG = 'en'


def process_image(image_path, enable_tts=True):
    """Process uploaded image"""
    if not image_path:
        return "Please upload an image", None
    
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set. Add it in Space secrets.", None
    
    try:
        from ocr import extract_text_from_image
        text = extract_text_from_image(Path(image_path), GROQ_API_KEY)
        
        if not text:
            return "No text found in image", None
        
        audio = None
        if enable_tts:
            from tts import text_to_speech
            audio = text_to_speech(text, TTS_LANG)
        
        return text, audio
    except Exception as e:
        return f"Error: {str(e)}", None


# Simple Gradio Interface
gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath", label="📷 Upload Book Page"),
        gr.Checkbox(label="🔊 Enable Text-to-Speech", value=True)
    ],
    outputs=[
        gr.Textbox(label="📝 Extracted Text", lines=8),
        gr.Audio(label="🔈 Audio")
    ],
    title="📚 Blind Reading Assistant",
    description="Upload a book page to extract text and generate audio"
).launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)