"""Python script with the modelling commands"""
import os
from dotenv import load_dotenv
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from loguru import logger

# load config values
load_dotenv()
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

def detect_and_draw_box(filename, model="yolov3-tiny", confidence=0.5):
    """Detects common objects on an image and creates a new image with bounding boxes.

    Args:
        filename (str): Filename of the image.
        model (str): Either "yolov3" or "yolov3-tiny". Defaults to "yolov3-tiny".
        confidence (float, optional): Desired confidence level. Defaults to 0.5.
    """
    
    # Images are stored under the images/ directory
    img_filepath = f'{RAW_DATA_PATH}/{filename}'
    
    # Read the image into a numpy array
    img = cv2.imread(img_filepath)
    if img.size > 0:
        logger.info(f'Sucessfully read image from {img_filepath}')
    else:
        logger.info(f'Image at {img_filepath} not loaded')
    
    # Perform the object detection
    bbox, label, conf = cv.detect_common_objects(img, confidence=confidence, model=model)
    
    # Print current image's filename
    logger.info(f"========================\nImage processed: {filename}\n")
    
    # Print detected objects with confidence level
    for l, c in zip(label, conf):
        print(f"Detected object: {l} with confidence level of {c}\n")
    
    # Create a new image that includes the bounding boxes
    output_image = draw_bbox(img, bbox, label, conf)

    output_filename = f'{OUTPUT_PATH}/boxed_{filename}'
    
    # Save the image in the directory images_with_boxes
    cv2.imwrite(output_filename, output_image)

    logger.info(f'Written image with boxes to {output_filename}')


if __name__ == "__main__":
    # run script
    # Some example images
    image_files = [
    'apple.jpg',
    'clock.jpg',
    'oranges.jpg',
    'car.jpg'
]
    # TODO add timer
    for image_file in image_files:
        detect_and_draw_box(image_file)

