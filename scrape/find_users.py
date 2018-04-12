import requests, json, re, time, urllib, time, csv
from bs4 import BeautifulSoup

base_url = "https://www.instagram.com/web/search/topsearch/?context=blended&query=%s"

accounts = []


def save_users_from_json(json, data):
	for user in json['users']: #runs through all users in search query
		if not user['user']['is_private']: #ensures that account is public
			if (user['user']['follower_count'] > 100000 and user['user']['follower_count'] < 200000): #pulls accounts who have 100k-200k followers
				data.append({
					"username": user['user']['username'],
					"followers": user['user']['follower_count']
				}) # appends these users' usernames to a list


### SINGLE LETTERS
for i in range(26): #runs through alphabet
	char = chr(97+i)
	print(char)
	time.sleep(2) #waits 2 seconds between requests to ensure we don't get blocked
	parsed = json.loads(requests.get(base_url % (char)).text) # Make the request and parse into JSON
	save_users_from_json(parsed, accounts) # Save users to accounts list


### DOUBLE LETTERS
for i in range(26): #runs through alphabet
	for j in range(26): #runs through alphabet again
		char = chr(97+i) + "" +  chr(97+j) #converts number to letter using ASCII to create a string with two letters
		print(char)
		time.sleep(2) #waits 2 seconds between requests to ensure we don't get blocked
		parsed = json.loads(requests.get(base_url % (char)).text) # Make the request and parse into JSON
		save_users_from_json(parsed, accounts) # Save users to accounts list


accounts = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in accounts)] # Removes duplicate dictionaries
#print(accounts)
print("Number of accounts: %d" % len(accounts))

# Save data to a csv.
with open('user_list.csv', 'w+') as file:
	for user in accounts:
		file.write(user['username'] + "," + str(user['followers']))
		file.write('\n')

print("Done")