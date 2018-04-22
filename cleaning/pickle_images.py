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


pickle_count = 0
pickle_limit = 42

# We pickle 1000 images per file, 42 files (42000 images total)
while pickle_count < pickle_limit:

	all_data = {
		
		"photos": [],
		"likes": [],
		"ratio": [],
		"url": [],
		"user": []
	}

	lower = pickle_count * 1000
	upper = (pickle_count + 1) * 1000
	print(lower, upper)
	for imgname in onlyfiles[lower:upper]:
		filepath = join(IMAGE_DATA_DIR, imgname)
		
		try:
			img = Image.open(filepath)
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
		except:
			print("Could not pickle: " + filepath)
			with open('pickle_error_log.txt', 'a') as f:
				f.write(filepath + '\n')


	pickle_count += 1
	print("Done with %d000" % pickle_count)
	pickle.dump(all_data, open("data_%s.pickle" & pickle_count, 'wb'))


#data = pickle.load(open("all_data.pickle", 'rb'))
#pprint.pprint(data)