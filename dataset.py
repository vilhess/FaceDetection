import torch
import albumentations as A
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import ast


transform = A.Compose([
    A.Resize(450, 450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
], bbox_params=A.BboxParams(format="albumentations", label_fields=['class_labels']))


class FaceDataset(Dataset):
    def __init__(self, partition='train', transform=transform):

        self.partition = partition
        self.transform = transform
        self.df = pd.read_csv(f"data/{partition}.csv")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        path_img = row['Unnamed: 0']
        img = cv2.imread(os.path.join('data/images', path_img))
        size = ast.literal_eval(row['size'])
        prev_coords = ast.literal_eval(row['coords'])
        coords = np.divide(prev_coords, [size[0], size[1], size[0], size[1]])
        transformed = self.transform(image=img,
                             bboxes=list(coords), 
                             class_labels=['face']*len(prev_coords))
        img_tr = transformed['image']
        bboxes_tr = transformed['bboxes']
        target = {}
        target['boxes'] = torch.tensor(bboxes_tr)
        target['labels'] = torch.ones(np.shape(bboxes_tr)[0], dtype = torch.int64)

        return torch.Tensor(img_tr).permute(2, 0, 1), target


def test():
    dataset = FaceDataset('train', transform)
    idx = np.random.choice(len(dataset))
    row = dataset[idx]
    img = row[0].permute(1, 2, 0)
    bbox = row[1]['boxes']
    for i in range(len(bbox)):
        c = bbox[i]
        c = np.multiply(c, [450, 450, 450, 450]).numpy().astype(int)
        cv2.rectangle(img.numpy(), c[:2], c[2:], (0, 255, 0), 2)
    plt.imshow(img.numpy())
    plt.show()
    plt.close()


if __name__=='__main__':
    test()