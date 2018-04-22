from PIL import Image
from os import listdir
from os.path import isfile, join, exists


PHOTO_DIR = 'photos'

onlyfiles = [f for f in listdir(PHOTO_DIR) if isfile(join(PHOTO_DIR, f))]

for file in onlyfiles:
	im=Image.open(join(PHOTO_DIR, file))
	print(im.size) # (width,height) tuple