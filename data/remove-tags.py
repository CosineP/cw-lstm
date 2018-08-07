#!/usr/bin/env python

# This was a one-off, not guaranteed to work anymore. Kept in source control
# for future reference.

# When I initially downloaded toots, I included the <tag>s in them. This uses a
# regex to remove them

import csv
import re

newline = re.compile(r'<br\s*?/?>')
paragraph = re.compile(r'</p><p>')
tags = re.compile(r'<.*?>')
codes = re.compile(r'&.*?;')

def sanitize(text):
	text = newline.sub('\n', text)
	text = paragraph.sub('\n\n', text)
	text = tags.sub('', text)
	text = codes.sub('', text)
	return text

with open('toots.csv', 'r') as r:
	read = csv.reader(r)
	with open('toots-sane.csv', 'w') as w:
		write = csv.writer(w)
		for row in read:
			# [content, cw]
			row[0] = sanitize(row[0])
			write.writerow(row)

