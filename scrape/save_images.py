import requests, json, re, time, urllib, csv, piexif, piexif.helper
from bs4 import BeautifulSoup


### Takes a username, and returns first 12 photos in a list of dictionaries.
def get_photos(username, followers):
	base = 'https://www.instagram.com/%s'
	user_id = ''
	after_id = ''
	post_count = 0
	vid_count = 0
	data = []

	# Retrieve JSON from endpoint
	soup = BeautifulSoup(requests.get(base % (username)).text, 'html.parser')
	sharedData = str(soup(text=re.compile(r'window._sharedData = '))[0])[:-1]
	parsed = json.loads(sharedData.replace('window._sharedData = ', ''))
	user = parsed['entry_data']['ProfilePage'][0]['graphql']['user']

	# Gather user info and data
	user_id = user['id']
	if user['edge_owner_to_timeline_media']['page_info']['has_next_page']:
		after_id = user['edge_owner_to_timeline_media']['page_info']['end_cursor']
	post_count = user['edge_owner_to_timeline_media']['count']
	nodes = user['edge_owner_to_timeline_media']['edges']

	# Grab first 12 posts
	for node in nodes:
		post = {
			"url": node['node']['thumbnail_resources'][4]['src'],
			"likes": node['node']['edge_liked_by'],
			"user": username,
			"ratio": node['node']['edge_liked_by']['count']/int(followers) # Likes to follower ratios
		}

		data.append(post)
		if node['node']['is_video']:
			vid_count += 1

	return data

# Get existing exif data of image file
def get_exif_data(image_file):
	try:
		exif_dict = piexif.load(image_file)
		return exif_dict
	except Exception as e:
		return 0

### Start saving data.

users = []

with open('user_list.csv', newline='') as csvfile:
	csv = csv.reader(csvfile, delimiter=',')
	for row in csv: # Cycle thru users
		user_posts = get_photos(row[0], row[1])
		counter = 1
		for post in user_posts:
			filename = 'photos/%s_%d.jpg' % (post['user'], counter)
			urllib.request.urlretrieve(post['url'], filename) # Save photos from URL
			time.sleep(2) # PREVENT US FROM GETTING BLOCKED
			#### WRITE METADATA
			exif_dict = get_exif_data(filename)
 
			if exif_dict is not 0:
				exif_dict['0th'][piexif.ImageIFD.Copyright] = "test"
				exif_dict['0th'][piexif.ImageIFD.Artist] = "test"
				exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(json.dumps(post))
			 
				exif_bytes = piexif.dump(exif_dict)
				piexif.insert(exif_bytes, filename)
			else:
				print("Was not able to open " + image)
			####
			counter += 1
		print("Finished %s" % row[0])