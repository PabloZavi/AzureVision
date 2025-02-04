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
    image_url="https://historia.nationalgeographic.com.es/medio/2020/01/22/churchill-con-su-famoso-sombrero-de-copa-en-una-foto-tomada-en-1945-cuando-la-segunda-guerra-mundial-llegaba-a-su-fin_73fb69b8_800x699.jpg",
    visual_features=[VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.READ,
            VisualFeatures.SMART_CROPS,
            VisualFeatures.PEOPLE,],
    language="en",
    gender_neutral_caption=True,  # Optional (default is False)
)

# Print all analysis results to the console
print("Image analysis results:")

if result.caption is not None:
    print(" Caption:")
    print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

if result.dense_captions is not None:
    print(" Dense Captions:")
    for caption in result.dense_captions.list:
        print(f"   '{caption.text}', {caption.bounding_box}, Confidence: {caption.confidence:.4f}")

if result.read is not None and result.read.blocks:
    print(" Read:")
    for line in result.read.blocks[0].lines:
        print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
        for word in line.words:
            print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")
else:
    print(" No text detected.")
    
if result.tags is not None:
    print(" Tags:")
    for tag in result.tags.list:
        print(f"   '{tag.name}', Confidence {tag.confidence:.4f}")

if result.objects is not None:
    print(" Objects:")
    for object in result.objects.list:
        print(f"   '{object.tags[0].name}', {object.bounding_box}, Confidence: {object.tags[0].confidence:.4f}")

if result.people is not None:
    print(" People:")
    for person in result.people.list:
        print(f"   {person.bounding_box}, Confidence {person.confidence:.4f}")

if result.smart_crops is not None:
    print(" Smart Cropping:")
    for smart_crop in result.smart_crops.list:
        print(f"   Aspect ratio {smart_crop.aspect_ratio}: Smart crop {smart_crop.bounding_box}")

print(f" Image height: {result.metadata.height}")
print(f" Image width: {result.metadata.width}")
print(f" Model version: {result.model_version}")