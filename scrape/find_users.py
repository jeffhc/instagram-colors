import requests, json, re, time, urllib
from bs4 import BeautifulSoup

base_url = "https://www.instagram.com/web/search/topsearch/?context=blended&query=%s"

accounts = []

for i in range(26): #runs through alphabet
	char = chr(97+i) #converts number to letter using ASCII
	parsed = json.loads(requests.get(base_url % (char)).text) # Make the request and parse into JSON
	for user in parsed['users']:
		if not user['user']['is_private']: #ensures that account is public
			if (user['user']['follower_count'] > 100000 and user['user']['follower_count'] < 200000): #pulls accounts who have 100k-200k followers
				#print(parsed['users'][j]['user']['username'] + " " + str(parsed['users'][j]['user']['follower_count']))
				accounts.append(user['user']['username'])

print(accounts)
print(len(accounts))
				