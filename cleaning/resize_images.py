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
	img = img.resize(INTENDED_SIZE, PIL.Image.ANTIALIAS)
	img.save(new_filepath)
	
	exif_dict = get_exif_data(new_filepath)
	 
	if exif_dict is not 0:
		exif_dict['0th'][piexif.ImageIFD.Copyright] = "test"
		exif_dict['0th'][piexif.ImageIFD.Artist] = "test"
		stuff = piexif.load(filepath)
		user_comment = piexif.helper.UserComment.load(stuff["Exif"][piexif.ExifIFD.UserComment])
		exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment)

				 
		exif_bytes = piexif.dump(exif_dict)
		piexif.insert(exif_bytes, new_filepath)
	else:
		print("Was not able to open " + filepath)
	print(counter)
	counter += 1
	with open('resize_info.txt', 'w') as file:
		file.write(str(file))