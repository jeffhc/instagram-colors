import requests, json, re, time, urllib, time, csv
from bs4 import BeautifulSoup

base_url = "https://www.instagram.com/web/search/topsearch/?context=blended&query=%s"

accounts = []

for i in range(26): #runs through alphabet
	for j in range(26): #runs through alphabet again
		char = chr(97+i) + "" +  chr(97+j) #converts number to letter using ASCII to create a string with two letters
		print(char)
		time.sleep(2) #waits 2 seconds between requests to ensure we don't get blocked
		parsed = json.loads(requests.get(base_url % (char)).text) # Make the request and parse into JSON

		for user in parsed['users']: #runs through all users in search query
			if not user['user']['is_private']: #ensures that account is public
				if (user['user']['follower_count'] > 100000 and user['user']['follower_count'] < 200000): #pulls accounts who have 100k-200k followers
					accounts.append(user['user']['username']) # appens these users' usernames to a list

accounts = list(set(accounts)) #converts list to set and back again to delete duplicates
print(accounts)
print("Number of accounts: " + len(accounts))