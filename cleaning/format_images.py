import pickle
import imageio
import numpy as np
from os import listdir 
from PIL import Image, ImageFilter
import os, os.path
from os.path import isfile, join, exists
import piexif, piexif.helper, json, pprint


IMAGE_DATA_DIR = 'scaled_photos'
data_count = len([name for name in os.listdir(IMAGE_DATA_DIR) if os.path.isfile(os.path.join(IMAGE_DATA_DIR, name))])

onlyfiles = [f for f in listdir(IMAGE_DATA_DIR) if isfile(join(IMAGE_DATA_DIR, f))]

#for images in range(len(onlyfiles)):
	#print(onlyfiles[images])

all_data = {
	
	"photos": [],
	"likes": [],
	"ratio": [],
	"url": [],
	"user": []
}

#pickle.dump(all_data, "doe.pickle")

for imgname in onlyfiles:
	img = Image.open(join(IMAGE_DATA_DIR,imgname))
	img.load()
	data = np.asarray( img, dtype="int32" )
	all_data["photos"].append(data)
	exif_dict = piexif.load(join(IMAGE_DATA_DIR, imgname))
	print(exif_dict)
	user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
	metadata = json.loads(user_comment)
	all_data["likes"].append(metadata["likes"])
	all_data["ratio"].append(metadata["ratio"])
	all_data["url"].append(metadata["url"])
	all_data["user"].append(metadata["user"])


pprint.pprint(all_data)

"""
all_data = {
	
	photos: [np.array(), np.array(), ],
	likes: []
}

pickle.dump(all_data, "doe.pickle")"""