import csv

users = []

with open('test_profiles.csv', newline='') as csvfile:
	csv = csv.reader(csvfile, delimiter=',')
	for row in csv:
		users.append(row[0])

