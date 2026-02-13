# AI Crop Insurance – ML Inference Script

This repository contains the core Machine Learning inference pipeline for our crop insurance solution.

## Features
- Crop classification (corn, cotton, paddy, wheat)
- Growth-stage detection
- Disease detection (blast, bacterial blight, healthy)
- Mandatory GPS geotag extraction from image EXIF
- Fertilizer, irrigation & pesticide advisory
- English/Hindi TTS (pyttsx3)
- JSON output for backend integrations
- Confidence-based invalid image detection

## Main File
- `farmer_full_inference3.py`  
  The complete inference script.

## Notes
- `.pth` model files are NOT uploaded to GitHub.
- Update model paths inside the script according to your local directory.
