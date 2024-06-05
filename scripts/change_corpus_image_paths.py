import json
import os

corpus_path = "/MarqoData/google_shopping/marqo-gs-dataset/marqo_gs_wfash_1m/corpus_1.json"
image_dir = "/MarqoData/google_shopping/images_wfash"

with open(corpus_path, "r") as f:
    corpus = json.load(f)

for key in corpus:
    corpus[key]['image_local'] = os.path.join(image_dir, os.path.basename(corpus[key]['image_local']))

with open(corpus_path, "w") as f:
    json.dump(corpus, f)