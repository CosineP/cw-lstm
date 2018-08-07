import re
from collections import Counter
import operator

with open('data/toots.csv') as f:
	text = f.read()
words = re.findall(r'\w+', text)
lowers = [word.lower() for word in words]
counts = Counter(lowers)
most = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

for i in range(500):
	print(most[i][0])

