import json
import os
import clip
import torch
from tqdm import tqdm

all_folders = os.listdir('pseudo_captions')
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
step = 8000
for a in all_folders:
    id_captions = json.load(open(os.path.join('pseudo_captions', a, 'id_captions.json'), 'r'))
    captions = list(id_captions.values())
    captions = [a[0] for a in captions]
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        all_text_features = []
        for i in tqdm(range(0, len(captions), step)):
            caption_text = clip.tokenize(captions[i:i+step]).to(device)
            caption_features = model.encode_text(caption_text)
            caption_features = caption_features.cpu()
            all_text_features.extend(caption_features)
        all_text_features = torch.stack(all_text_features)
        torch.save(caption_features, os.path.join('pseudo_captions', a, 'caption_features.pt'))