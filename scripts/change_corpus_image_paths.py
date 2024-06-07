import json
import os
import sys

corpus_path = sys.argv[1]
image_dir = sys.argv[2]

with open(corpus_path, "r") as f:
    corpus = json.load(f)

for key in corpus:
    corpus[key]['image_local'] = os.path.join(image_dir, os.path.basename(corpus[key]['image_local']))

with open(corpus_path, "w") as f:
    json.dump(corpus, f)
