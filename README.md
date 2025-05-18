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
or 

```bash
pip install uv
```

2. Create and activate a virtual environment:
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -r pyproject.toml
```

## Usage

1. Place your symbol image as `resonator.png` in the script's directory
2. Create a `data` directory and place the images you want to analyze inside it
3. Run the script:
```bash
python main.py
```

The results will be saved in a `results` directory.

## ⚠️ Warning

The template matching algorithm works only when searching for the exact symbol from which the template was extracted.