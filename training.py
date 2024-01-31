import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from tqdm import tqdm
from dataset import FaceDataset

NUM_CLASSES = 2
BATCH_SIZE = 4
INIT_EPOCH = 15
EPOCHS = 20
DEVICE = 'cpu'


def custom_collate(data):
    return data

weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model=fasterrcnn_resnet50_fpn(weights=weights)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

classification_head=list(model.children())[-2:]

for children in list(model.children())[:-2]:
    for params in children.parameters():
        params.requires_grad=False
        
parameters=[]
for heads in classification_head:
    for params in heads.parameters():
        parameters.append(params)

optimizer = torch.optim.Adam(parameters, lr = 3e-4)

if INIT_EPOCH-1>=0:
    optimizer.load_state_dict(torch.load(f'checkpoints/epoch_{INIT_EPOCH-1}.pth')['optimizer_state_dict'])
    model.load_state_dict(torch.load(f'checkpoints/epoch_{INIT_EPOCH-1}.pth')['model_state_dict'])
model = model.to(DEVICE)

trainset = FaceDataset()
trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

def training():
    for epoch in range(INIT_EPOCH, EPOCHS):
        epoch_loss = 0
        tqdm_iterator = tqdm(trainloader, total = len(trainloader))
        for data in tqdm_iterator:
            imgs = []
            targets = []
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        checkpoint = {'EPOCH':epoch,
                      'model_state_dict':model.state_dict(),
                      'optimizer_state_dict':optimizer.state_dict(),
                      'LOSS':epoch_loss}
        print(f'Loss for epoch {epoch} is {epoch_loss}')
        torch.save(checkpoint, f'checkpoints/epoch_{epoch}.pth')



if __name__ == '__main__':
    training()