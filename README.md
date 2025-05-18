# Template Matching Script

This Python script performs template matching to find a specific symbol (resonator) in multiple images. It can detect the symbol at different rotation angles and marks the detected locations on the output images.

## Features

- Detects a symbol in multiple images
- Supports symbol detection at different rotation angles (0-360 degrees)
- Handles multiple image formats (PNG, JPG, JPEG, GIF)
- Applies non-maximum suppression to avoid duplicate detections
- Saves results with detection boxes and rotation angles

## Setup Development Environment

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
```

3. Install dependencies:
```bash
uv pip install opencv-python numpy matplotlib loguru
```

## Usage

1. Place your symbol image as `resonator.png` in the script's directory
2. Create a `data` directory and place the images you want to analyze inside it
3. Run the script:
```bash
python main.py
```

The results will be saved in a `results` directory.

## ⚠️ Important Warning

The template matching algorithm works best when searching for the exact symbol from which the template was extracted. The accuracy may significantly decrease when trying to detect similar but not identical symbols. For optimal results:

1. Use the exact same symbol as your template
2. Ensure consistent image quality and resolution
3. Be aware that variations in lighting, scale, or perspective might affect detection accuracy
