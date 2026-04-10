#!/usr/bin/env python3
"""
OCR Module - Using Groq API for Vision
"""

import base64
from pathlib import Path
from typing import Optional

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


def encode_image(image_path: Path) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def extract_text_from_image(image_path: Path, api_key: str, prompt: Optional[str] = None) -> str:
    """Extract text from image using Groq API"""
    if not HAS_GROQ:
        raise Exception("Groq library not installed. Run: pip install groq")
    
    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        raise Exception("GROQ_API_KEY not configured. Please set your API key in config.py")
    
    if prompt is None:
        prompt = "Extract ALL text from this book page image. Return only the text content."
    
    base64_image = encode_image(image_path)
    
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.3,
            max_completion_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"OCR failed: {str(e)}")


def extract_text_simple(image_path: Path, api_key: str) -> str:
    """Simple version"""
    return extract_text_from_image(image_path, api_key)