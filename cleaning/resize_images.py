import PIL
from PIL import Image
from os import listdir, makedirs
from os.path import isfile, join, exists
import os

RAW_PHOTO_DIR = '../scrape/photos'
SCALED_PHOTO_DIR = 'scaled_photos'
INTENDED_SIZE = (640, 640)


# Get filenames in the raw photo directory
onlyfiles = [f for f in listdir(RAW_PHOTO_DIR) if isfile(join(RAW_PHOTO_DIR, f))]

# Check if scaled_photos directory exists
if not os.path.exists(SCALED_PHOTO_DIR):
    os.makedirs(SCALED_PHOTO_DIR)

# Resize and save images
for file in onlyfiles:
	img = Image.open(join(RAW_PHOTO_DIR, file))
	img = img.resize(INTENDED_SIZE, PIL.Image.ANTIALIAS)
	img.save(join(SCALED_PHOTO_DIR, file[:-4] + "_r.jpg"))