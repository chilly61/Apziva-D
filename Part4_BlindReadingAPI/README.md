---
title: MonReader
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Blind Reading Assistant

Upload book page → Extract text → Generate audio

## Setup
Set GROQ_API_KEY in Space secrets (from https://console.groq.com/keys)

## Tech
- OCR: Groq API (Llama-4-Scout Vision)
- TTS: gTTS (free)
- Host: HuggingFace Spaces (Free)