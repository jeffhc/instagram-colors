import requests, json, re, time, urllib
from bs4 import BeautifulSoup


username = 'elonmusk'
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
	data.append(node['node']['display_url'])
	if node['node']['is_video']:
		vid_count += 1

# Grab the rest of the posts
"""data_url = 'https://www.instagram.com/graphql/query/?query_hash=42323d64886122307be10013ad2dcc44&variables='
variables = '{"id":"%s","first":%d,"after":%s}' % (user_id, post_count - 12, after_id)
data_url += urllib.parse.quote(str(variables))
parsed = requests.get(data_url).text
print(data_url)
nodes = parsed['data']['user']['edge_owner_to_timeline_media']['edges']
for node in nodes:
	data.append(node['node']['display_url'])
	if node['node']['is_video']:
		vid_count += 1"""

print('Post number verification: %d' % post_count)
print('Data list size: %d' % len(data))
print('Vid count: %d' % vid_count)

# Write data to file
with open('ig_%s.txt' % username, 'w') as file:
	for item in data:
		file.write(json.dumps(item) + '\n')