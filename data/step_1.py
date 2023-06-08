import json
import nltk
from tqdm import tqdm

### load ShapeNet Nouns ###
taxonomy = json.load(open('taxonomy.json', 'r'))
synset_list = json.load(open('synset_list.json', 'r'))
used_class = list(synset_list.values())[:6]
noun_v = []
hierarchy = {a: [a] for a in used_class}
for k, v in hierarchy.items():
    for a in taxonomy:
        if a['synsetId'] in v or a['synsetId'] == k:
            v.extend(a['children'])

noun_dict = {}
for k, v in hierarchy.items():
    noun_dict[k] = []
    for idx in v:
        for a in taxonomy:
            if a['synsetId'] == idx:
                noun_dict[k].extend(a['name'].split(','))
                break
json.dump(noun_dict, open('noun_shapenet.json', 'w'))

### load CLIP tokens ###
clip_tokens = json.load(open('clip_tokens.json', 'r'))
clip_tokens = [a[:-4] if '</w>' in a else a for a in clip_tokens]
clip_tokens = list(set(clip_tokens))
adjectives = ['JJ', 'JJR', 'JJS']
nouns = ['NN', 'NNS']
vocab = {'adj': [], 'noun': []}
for a in tqdm(clip_tokens):
    words = nltk.word_tokenize(a)
    pos = nltk.pos_tag(words)
    for b in pos:
        if b[0].encode('UTF-8').isalpha():
            if b[1] in adjectives:
                vocab['adj'].append(b[0])
            if b[1] in nouns:
                vocab['noun'].append(b[0])
print(vocab)
json.dump(vocab, open('vocab_clip.json', 'w'))
