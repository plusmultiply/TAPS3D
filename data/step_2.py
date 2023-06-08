import json
import clip
import torch.utils.data as data
import os
import cv2
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
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

        img_index = self.img_list[index].split('/')[-2]

        return self.image_transform(image), img_index, self.img_list[index]

@click.command()
@click.option('--render_path', help='Path to the rendered 2D images', default='/mnt/bd/text-3d/get3d/data/ShapeNetCore_save/img', required=True)

def main(render_path):
    vocab_clip = json.load(open('vocab_clip.json', 'r'))
    vocab_clip['adj'].remove('simulated')
    synset_list = json.load(open('synset_list.json', 'r'))
    shapenet_name = list(synset_list.keys())
    shapenet_id = list(synset_list.values())
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    colors = ['red', 'green', 'blue', 'yellow', 'brown', 'purple', 'white', 'black']


    for i, id_ in enumerate(shapenet_id[:6]):
        training_set = ImageDataset(os.path.join(render_path, id_), preprocess) 
        dataloader = torch.utils.data.DataLoader(
                dataset=training_set, batch_size=16,
                num_workers=8, shuffle=False)
        noun_shapenet = json.load(open('noun_shapenet.json', 'r'))
        noun_shapenet = noun_shapenet[id_]
        class_name = shapenet_name[i].split('_')[1]
        step = 200
        with torch.no_grad():
            text = clip.tokenize(noun_shapenet).to(device)
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            all_adj_text_features = []
            for i in tqdm(range(0, len(vocab_clip['adj']), step)):
                adj_text = clip.tokenize(vocab_clip['adj'][i:i+step]).to(device)
                adj_text_features = model.encode_text(adj_text).cpu()
                all_adj_text_features.extend(adj_text_features)
            all_adj_text_features = torch.stack(all_adj_text_features)
            all_adj_text_features = all_adj_text_features / all_adj_text_features.norm(dim=1, keepdim=True)
            all_noun_text_features = []
            for i in tqdm(range(0, len(vocab_clip['noun']), step)):
                noun_text = clip.tokenize(vocab_clip['noun'][i:i+step]).to(device)
                noun_text_features = model.encode_text(noun_text).cpu()
                all_noun_text_features.extend(noun_text_features)
            all_noun_text_features = torch.stack(all_noun_text_features)
            all_noun_text_features = all_noun_text_features / all_noun_text_features.norm(dim=1, keepdim=True)
            color_text = clip.tokenize(colors).to(device)
            color_text_features = model.encode_text(color_text).cpu()

        K = 3
        noun_shapenet = np.array(noun_shapenet)

        all_retrieved_word = []
        all_retrieved_adj = []
        all_index = []
        for (images, index, img_path) in tqdm(dataloader):
            images = images.to(device)
            with torch.no_grad():
                image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            # cosine similarity as logits
            shapenet_logits = image_features @ text_features.t()
            values, idx = shapenet_logits.topk(K)
            retrieved_word = noun_shapenet[idx.cpu().numpy()]
            all_retrieved_word.append(retrieved_word)

            adj_logits = image_features.cpu().float() @ all_adj_text_features.t().float()
            values, idx = adj_logits.float().topk(K*2)
            retrieved_adj = np.array(vocab_clip['adj'])[idx.numpy()]
            color_logits = image_features.cpu().float() @ color_text_features.t().float()
            values, idx = color_logits.float().topk(1)
            retrieved_colors = np.array(colors)[idx.numpy()]
            retrieved_adj = np.concatenate([retrieved_adj, retrieved_colors], 1)

            all_retrieved_adj.append(retrieved_adj)
            all_index.extend(list(index))

        all_retrieved_word = np.concatenate(all_retrieved_word)
        all_retrieved_adj = np.concatenate(all_retrieved_adj)
        path = os.path.join('pseudo_captions', class_name)
        if not os.path.exists(path):
            os.makedirs(path) 
        json.dump(all_index, open(os.path.join(path, 'index.json'), 'w'))
        np.save(os.path.join(path, 'retrieved_word.npy'), all_retrieved_word)
        np.save(os.path.join(path, 'retrieved_adj.npy'), all_retrieved_adj)

if __name__ == "__main__":
    main()