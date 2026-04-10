#!/usr/bin/env python3
"""
TTS Module - Using gTTS (free, no GPU needed)
"""

import tempfile
from pathlib import Path
from gtts import gTTS


def generate_speech(text: str, output_path: Path, lang: str = "en") -> Path:
    """Generate speech using gTTS"""
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(str(output_path))
    return output_path


def text_to_speech(text: str, lang: str = "en") -> bytes:
    """Convert text to speech and return audio bytes"""
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Save to temp file then read bytes
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name
    
    tts.save(temp_path)
    
    with open(temp_path, "rb") as f:
        audio_bytes = f.read()
    
    # Clean up
    Path(temp_path).unlink(missing_ok=True)
    
    return audio_bytes