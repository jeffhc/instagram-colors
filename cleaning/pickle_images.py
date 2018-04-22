import pickle
import imageio
import numpy as np
from os import listdir 
from PIL import Image, ImageFilter
import os, os.path
from os.path import isfile, join, exists
import piexif, piexif.helper, json, pprint


IMAGE_DATA_DIR = 'scaled_photos'
onlyfiles = [f for f in listdir(IMAGE_DATA_DIR) if isfile(join(IMAGE_DATA_DIR, f))]


all_data = {
	
	"photos": [],
	"likes": [],
	"ratio": [],
	"url": [],
	"user": []
}

for imgname in onlyfiles:
	filepath = join(IMAGE_DATA_DIR, imgname)
	
	img = Image.open(imgname)
	img.load()
	data = np.asarray( img, dtype="int32" )
	all_data["photos"].append(data)

	exif_dict = piexif.load(filepath)
	user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
	metadata = json.loads(user_comment)
	
	all_data["likes"].append(metadata["likes"])
	all_data["ratio"].append(metadata["ratio"])
	all_data["url"].append(metadata["url"])
	all_data["user"].append(metadata["user"])



pickle.dump(all_data, open("all_data.pickle", 'wb'))


#data = pickle.load(open("all_data.pickle", 'rb'))
#pprint.pprint(data)