import os
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
try:
    load_dotenv()
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Create an Image Analysis client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# Image URL
image_url = "https://www.techsmith.es/blog/wp-content/uploads/2022/12/jpg-png.png"

# Analyze image
result = client.analyze_from_url(
    image_url=image_url,
    visual_features=[VisualFeatures.READ],  # Only OCR
    language="en"
)

# Download the image
response = requests.get(image_url, stream=True)
image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Convert BGR to RGB for proper visualization
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process OCR results
if result.read is not None and result.read.blocks:
    for block in result.read.blocks:
        for line in block.lines:
            # Get bounding box points
            points = line.bounding_polygon
            pts = np.array([[p.x, p.y] for p in points], np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Draw bounding box
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # Put text on the image
            x, y = pts[0][0]
            cv2.putText(image, line.text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the image
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.show()
