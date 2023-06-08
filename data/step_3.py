import json
from text_templates import imagenet_templates, part_templates, imagenet_templates_small
import numpy as np
import itertools as it
import clip
import torch.utils.data as data
import cv2
from PIL import Image
import os
import torch
from tqdm import tqdm
import random
import click

class ImageDataset(data.Dataset):
    def __init__(self, image_root, image_transform):
        seed = 0

        self.img_list = []
        for root, dirs, files in os.walk(image_root):
            if files != []:
                files = [os.path.join(root, a) for a in files]
                self.img_list.append(files[np.random.RandomState(seed).randint(len(files))])
        self.img_list.sort()

        self.image_transform = image_transform

    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, index):
        ori_img = cv2.imread(self.img_list[index], cv2.IMREAD_UNCHANGED)
        img = ori_img[:, :, :3][..., ::-1]
        image = Image.fromarray(img)

        class_id = self.img_list[index].split('/')[-3]
        img_index = self.img_list[index].split('/')[-2]

        return self.image_transform(image), img_index, class_id, self.img_list[index]

@click.command()
@click.option('--render_path', help='Path to the rendered 2D images', default='/mnt/bd/text-3d/get3d/data/ShapeNetCore_save/img', required=True)

def main(render_path):
    
    seed = 0
    synset_list = json.load(open('synset_list.json', 'r'))
    shapenet_name = list(synset_list.keys())
    shapenet_id = list(synset_list.values())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    template = '{} {}.'
    template3 = '{} {} {}.'

    for i, id_ in enumerate(shapenet_id[:6]):
        class_name = shapenet_name[i].split('_')[1]
        path = os.path.join('pseudo_captions', class_name)

        all_retrieved_word = np.load(os.path.join(path, 'retrieved_word.npy'), allow_pickle=True)
        all_retrieved_adj = np.load(os.path.join(path, 'retrieved_adj.npy'), allow_pickle=True)
        all_index = json.load(open(os.path.join(path, 'index.json'), 'r'))
        
        training_set = ImageDataset(os.path.join(render_path, id_), preprocess) 
        dataloader = torch.utils.data.DataLoader(
                dataset=training_set, batch_size=1,
                num_workers=1, shuffle=False)

        all_captions = []
        id_captions = {}
        class_ = 0
        for (images, index, class_id, img_path) in tqdm(dataloader):
            images = images.to(device)
            with torch.no_grad():
                image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            idx = all_index.index(index[0])

            retrieved_word = all_retrieved_word[idx]
            retrieved_adj = all_retrieved_adj[idx]

            all_fake_captions = []
            concepts = []
            for i in range(len(list(retrieved_adj))):
                for k in range(len(retrieved_word)):
                    generated_caption = template.format(retrieved_adj[i], retrieved_word[k])
                    all_fake_captions.append(generated_caption)
                    concepts.append([retrieved_adj[i], retrieved_word[k]])

            with torch.no_grad():
                text = clip.tokenize(all_fake_captions).to(device)
                text_features = model.encode_text(text)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logits = image_features @ text_features.t()
            selected_idx = logits.argmax()
            selected_idx = selected_idx.cpu()
            if np.random.random() < 0.2 or concepts[selected_idx][0] == retrieved_adj[-1]:  ## without color
                fake_captions = all_fake_captions[selected_idx]
            else:  ## with color
                fake_captions = template3.format(retrieved_adj[-1], concepts[selected_idx][0], concepts[selected_idx][1])
            id_captions[index[0]] = [fake_captions, class_id[0]]
            all_captions.extend(random.sample(all_fake_captions, 20))
        json.dump(all_captions, open(os.path.join(path, 'all_captions.json'), 'w'))
        json.dump(id_captions, open(os.path.join(path, 'id_captions.json'), 'w'))


if __name__ == "__main__":
    main()