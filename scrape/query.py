import requests, json, re, time, urllib
from bs4 import BeautifulSoup

base_url = "https://www.instagram.com/web/search/topsearch/?context=blended&query=%s"
query = 'a'

# Make the request and parse into JSON
parsed = json.loads(requests.get(base_url % (query)).text)

print(parsed['users'][0]['user']['username'])

rank = parsed['rank_token']


parsed = json.loads(requests.get(base_url % (query) + '&rank_token=%s' % rank).text)
print(parsed['users'][0]['user']['username'])
