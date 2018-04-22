import PIL
from PIL import Image
from os import listdir, makedirs
from os.path import isfile, join, exists
import os
import piexif, piexif.helper, json, pprint

RAW_PHOTO_DIR = '../scrape/photos'
SCALED_PHOTO_DIR = 'scaled_photos'
INTENDED_SIZE = (640, 640)

def get_exif_data(image_file):
	try:
		exif_dict = piexif.load(image_file)
		return exif_dict
	except Exception as e:
		return 0


# Get files in the raw photo directory
onlyfiles = [f for f in listdir(RAW_PHOTO_DIR) if isfile(join(RAW_PHOTO_DIR, f))]

# Check if scaled_photos directory exists
if not os.path.exists(SCALED_PHOTO_DIR):
    os.makedirs(SCALED_PHOTO_DIR)

# Resize and save images
counter = 0
for file in onlyfiles:
	filepath = join(RAW_PHOTO_DIR,file)
	new_filepath = join(SCALED_PHOTO_DIR, file[:-4] + "_r.jpg")
	img = Image.open(filepath)
	if (img.size) != INTENDED_SIZE:
		img = img.resize(INTENDED_SIZE, PIL.Image.ANTIALIAS)
	img.save(new_filepath)
	
	
	new_metadata = get_exif_data(new_filepath)
	if new_metadata is not 0:
		# Read metadata from old image
		old_metadata = piexif.load(filepath)
		user_comment = piexif.helper.UserComment.load(old_metadata["Exif"][piexif.ExifIFD.UserComment])
		# Write metadata to new image
		new_metadata["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment)
		exif_bytes = piexif.dump(new_metadata)
		piexif.insert(exif_bytes, new_filepath)
	else:
		print("Was not able to open " + new_filepath)

	print(counter)
	counter += 1
	with open('resize_info.txt', 'w') as f:
		f.write(file)