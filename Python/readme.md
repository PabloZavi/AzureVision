# Vision Analysis Python Scripts

This repository contains three Python scripts for analyzing and processing images using various visual features provided by the Azure AI Vision SDK.

## 1. vision.py

**Functionality:**
- Adds a caption to an image.
- If there is text in the image, shows the text and its coordinates.

## 2. vision2.py

**Functionality:**
- Similar to `vision.py`, but with additional tools:
  - `VisualFeatures.TAGS`: Provides high-level tags describing the image content.
  - `VisualFeatures.OBJECTS`: Identifies and locates specific objects within the image.
  - `VisualFeatures.CAPTION`: Generates a brief textual description of the image.
  - `VisualFeatures.DENSE_CAPTIONS`: Provides detailed descriptions of different parts of the image.
  - `VisualFeatures.READ`: Reads and extracts text from the image.
  - `VisualFeatures.SMART_CROPS`: Suggests smart crops to improve the visual composition.
  - `VisualFeatures.PEOPLE`: Detects and provides information about people in the image.

## 3. VisionOCR.py

**Functionality:**
- Draws rectangles and text annotations over the detected words in a fetched image.
- If text is found, displays the image with red bounding boxes and text labels.
- If no text is detected, simply displays the original image.
