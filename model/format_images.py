import pickle
import imageio
import numpy as np
from PIL import Image, ImageFilter
import os, os.path


IMAGE_DATA_DIR = '../scrape/photos'
data_count = print(len([name for name in os.listdir(IMAGE_DATA_DIR) if os.path.isfile(os.path.join(IMAGE_DATA_DIR, name))]))

all_data = {
	
	photos: [np.array(), np.array(), ],
	likes: []
}

pickle.dump(all_data, "doe.pickle")