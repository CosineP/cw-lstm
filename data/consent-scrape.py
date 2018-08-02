#!/usr/bin/env python

# Do scraping. Only people who have consented.
# https://anticapitalist.party/@cosine/100478734475112961

from mastodon import Mastodon
import csv

def register():
	# Register app - only once!
	Mastodon.create_app(
		 'consent-scrape',
		 api_base_url = 'https://anticapitalist.party',
		 to_file = 'mastoclient.secret'
	)

def log_in():
	# Log in - either every time, or use persisted
	mastodon = Mastodon(
		client_id = 'mastoclient.secret',
		api_base_url = 'https://anticapitalist.party'
	)
	username = input('Your username: ')
	password = input('Your password: ')
	mastodon.log_in(
		username,
		password,
		to_file = 'mastologin.secret'
	)

def toots(handle):
	search_result = mastodon.account_search(handle)[0]
	max_id = None
	toots = True
	while toots:
		toots = mastodon.account_statuses(search_result, max_id=max_id)
		for toot in toots:
			if toot.visibility == 'public':
				yield toot
			max_id = toot.id

def get_already():
	with open('already') as f:
		ids = f.readlines()
		return set(ids)

def check_already(already, toot_id):
	if (str(toot_id)+'\n') in already:
		return True
	else:
		already_file.write(str(toot.id) + '\n')
		return False

try:
	# Create actual API instance
	mastodon = Mastodon(
		access_token = 'mastologin.secret',
		api_base_url = 'https://anticapitalist.party',
		ratelimit_method='pace',
	)
except:
	register()
	log_in()
	print("Run program again now that you've logged in")
	exit()

consented = []
with open('consented.txt') as f:
	consented = f.readlines()

print("initializing what IDs we've already downloaded...")
already = get_already()

already_file = open('already', 'a')

with open('toots.csv', 'a') as f:
	write = csv.writer(f)
	already_count = 0
	already_tolerance = 400
	for handle in consented:
		count = 0
		handle = handle.strip()
		print(handle)
		for toot in toots(handle):
			count += 1
			if check_already(already, toot.id):
				already_count += 1
				print("Skipping already seen toot " + str(toot.id))
				if already_count > already_tolerance:
					print("Skipping account " + handle + " due to many alreadys")
					break
			else:
				# Decrement the already_count so it's kinda like consecutive
				# but not that aggressive. This makes it so a few common boosts
				# doesn't kill it
				# The max just keeps it >=0
				already_count = max(already_count-1, 0)
			content = toot.content
			cw = toot.spoiler_text
			pair = [content, cw]
			write.writerow(pair)
			print("\r%d toots downloaded..." % count, end='', flush=True)
		# Keep the last count of toots downloaded with a final print
		print("Finished downloading this person's toots")

