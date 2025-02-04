import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
try:
    load_dotenv()
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Create an Image Analysis client for synchronous operations,
# using API key authentication
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# Get a caption for the image. This will be a synchronously (blocking) call.
result = client.analyze_from_url(
    image_url="https://badac.uniandes.edu.co/ramirezvillamizar/files/fullsize/8a17acfaf99da41d620f2fd06640f8b7.jpg",
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    language="en",
    gender_neutral_caption=True,  # Optional (default is False)
)

print("Image analysis results:")
# Print caption results to the console
if result.caption is not None:
    print(" Caption:")
    print(f"   '{result.caption.text}', Confidence {
          result.caption.confidence:.4f}")
else:
    print(" No caption detected.")

# Print text (OCR) analysis results to the console
full_text = ""
if result.read is not None and result.read.blocks:
    print(" Read:")
    for line in result.read.blocks[0].lines:
        full_text += line.text + '\n' 
        print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
        for word in line.words:
            print(f"     Word: '{word.text}', Bounding polygon {
                  word.bounding_polygon}, Confidence {word.confidence:.4f}")
else:
    print(" No text detected.")

# Print the full text
if full_text != "":
    print("Full text: \n", full_text)