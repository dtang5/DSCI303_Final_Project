import json
filename = "review.json"
stars, reviews = [], []
with open(filename, 'r') as f:
    for line in f:
        data = json.loads(line)
        stars.append(data["stars"])
        reviews.append(data["text"])
    f.close()
to_write = zip(reviews, stars)

with open("reviews_to_stars.txt", 'w') as f1:
    for tup in to_write:
        f1.write(str(tup))
    f1.close