import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from tqdm import tqdm
from dataset import FaceDataset


NUM_CLASSES = 2
DEVICE = 'cpu'


weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model=fasterrcnn_resnet50_fpn(weights=weights)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model = model.to(DEVICE)

testset = FaceDataset('test')

def test():
    testloader = DataLoader(testset, batch_size=4, shuffle=False, collate_fn=lambda data: data)
    for e in range(3):
        model.load_state_dict(torch.load(f'checkpoints/epoch_{e}.pth')['model_state_dict'])
        epoch_loss = 0
        imgs = []
        targets = []
        tqdm_iterator = tqdm(testloader, total = len(testloader))
        for data in tqdm_iterator:
            for d in data:
                imgs.append(d[0].to(DEVICE))
                targ = {}
                targ['boxes'] = d[1]['boxes'].to(torch.float16).to(DEVICE)
                targ['labels'] = d[1]['labels'].to(DEVICE)
                targets.append(targ)
            loss_dict = model(imgs, targets)
            loss = sum(v for v in loss_dict.values())
            tqdm_iterator.set_postfix({"Current loss": loss.cpu().detach().numpy()})
            epoch_loss+=loss.cpu().detach().numpy()
        print(f"Test loss on epoch {e} is {epoch_loss}")




def show_inference(min_score, epoch):
    model.load_state_dict(torch.load(f'checkpoints/epoch_{epoch}.pth')['model_state_dict'])
    sample = np.random.choice(len(testset), 4, replace=False)
    fig = plt.figure()
    model.eval()
    for i, ech in enumerate(sample):
        ax = fig.add_subplot(2, 2, i+1)
        img, label = testset[i]
        preds = model(img.to(DEVICE).unsqueeze(0))
        for j in range(len(preds[0]['boxes'])):
            if preds[0]['scores'][j].detach()>=min_score:
                coords = preds[0]['boxes'][j].detach()
                coords = np.multiply(coords, [450, 450, 450, 450])
                cv2.rectangle(img.permute(1, 2, 0).numpy(), coords[:2].numpy().astype(int), coords[2:].numpy().astype(int), (250, 0, 0), 2)
        ax.imshow(img.permute(1, 2, 0).numpy())
    plt.show()
    plt.close()


if __name__=='__main__':
    show_inference(0.3, 9)
    # test()