import requests, json, re, time
from bs4 import BeautifulSoup

user = 'dude90'

base = 'https://www.instagram.com/%s/?max_id=%s'

has_more = True
max_id = ''
data = []
posts = 0
vid_count = 0

while has_more:
	soup = BeautifulSoup(requests.get(base % (user, max_id)).text, 'html.parser')
	sharedData = str(soup(text=re.compile(r'window._sharedData = '))[0])[:-1]
	parsed = json.loads(sharedData.replace('window._sharedData = ', ''))
	
	if 'ProfilePage' in parsed['entry_data']:
		nodes = parsed['entry_data']['ProfilePage'][0]['graphql']['user']['edge_owner_to_timeline_media']['edges']
	else:
		print("BREAKING")
		break

	posts += len(nodes)
	for node in nodes:
		data.append(node['node']['display_url'])
		#print(node['node']['display_url'])
		if node['node']['is_video']:
			vid_count += 1
	
	next_page = parsed['entry_data']['ProfilePage'][0]['graphql']['user']['edge_owner_to_timeline_media']['page_info']
	if(next_page['has_next_page']):
		max_id = next_page['end_cursor']
		print(max_id)
	else:
		has_more = False

	# Timer for rate limiting
	time.sleep(2)

print('Post number verification: %d' % posts)
print('Vid count: %d' % vid_count)

# WRITE DATA TO FILE
with open('ig_%s.txt' % user, 'w') as file:
	for item in data:
		file.write(json.dumps(item) + '\n')