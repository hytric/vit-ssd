import torch
from torch import nn
import d2l

from torch.utils.data import Dataset
from PIL import Image
import os

from torchvision import transforms
from torch.utils.data import DataLoader

import json


class CustomImageDataset(Dataset):
    '''
    JSON 파일의 구조
        {
            "image_001.jpg": {"boxes": [[x1, y1, x2, y2]], "labels": [0]},
            "image_002.jpg": {"boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2]], "labels": [0, 1]},
            ...
        }
    '''
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform
        with open(label_file, 'r') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        boxes = self.labels[self.img_names[idx]]['boxes']
        labels = self.labels[self.img_names[idx]]['labels']

        # bounding box 좌표를 이미지의 크기에 맞게 복원
        w, h = image.size
        boxes = [[x1 * w, y1 * h, x2 * w, y2 * h] for x1, y1, x2, y2 in boxes]

        return image, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class TinySSD(nn.Module):
    def __init__(self, num_classes):
        super(TinySSD, self).__init__()
        self.num_classes = num_classes
        for i in range(5):
            setattr(self, f'blk_{i}', self.get_blk(i))
            setattr(self, f'cls_{i}', self.cls_predictor(128, num_anchors, num_classes))
            setattr(self, f'bbox_{i}', self.bbox_predictor(128, num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = self.blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = self.concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = self.concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

    def cls_predictor(self, num_inputs, num_anchors, num_classes):
        return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

    def bbox_predictor(self, num_inputs, num_anchors):
        return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

    def get_blk(self, i):
        if i == 0:
            blk = self.base_net()
        elif i == 1:
            blk = self.down_sample_blk(64, 128)
        elif i == 4:
            blk = nn.AdaptiveMaxPool2d((1,1))
        else:
            blk = self.down_sample_blk(128, 128)
        return blk

    def blk_forward(self, X, blk, size, ratio, cls_predictor, bbox_predictor):
        Y = blk(X)
        anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
        cls_preds = cls_predictor(Y)
        bbox_preds = bbox_predictor(Y)
        return (Y, anchors, cls_preds, bbox_preds)

    def concat_preds(self, preds):
        return torch.cat([self.flatten_pred(p) for p in preds], dim=1)

    def flatten_pred(self, pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


class Trainer():
    def __init__(self, net, data_loader, device):
        self.net = net
        self.data_loader = data_loader
        self.device = device

    def train(self, num_epochs):
        self.net = self.net.to(self.device)
        trainer = torch.optim.SGD(self.net.parameters(), lr=0.2, weight_decay=5e-4)
        cls_loss = nn.CrossEntropyLoss(reduction='none')
        bbox_loss = nn.L1Loss(reduction='none')

        for epoch in range(num_epochs):
            self.net.train()
            for features, target in self.data_loader:
                trainer.zero_grad()
                X, Y = features.to(self.device), target.to(self.device)
                anchors, cls_preds, bbox_preds = self.net(X)
                bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, cls_loss, bbox_loss)
                l.mean().backward()
                trainer.step()

        # 모든 학습이 끝난 후에 모델의 파라미터를 저장합니다.
        torch.save(self.net.state_dict(), 'model_final.pth')

    @staticmethod
    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, cls_loss, bbox_loss):
        batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
        cls = cls_loss(cls_preds.reshape(-1, num_classes),
                       cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = bbox_loss(bbox_preds * bbox_masks,
                         bbox_labels * bbox_masks).mean(dim=1)
        return cls + bbox


if __name__ == "__main__":
    # 이미지 전처리를 위한 transform 생성
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 이미지 크기 조절
        transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환
    ])

    # 커스텀 이미지 데이터셋 로드
    dataset = CustomImageDataset('./path/to/your/images', './path/to/your/labels.json', transform=transform)

    # DataLoader 생성
    train_iter = DataLoader(dataset, batch_size=32, shuffle=True)


    # 학습
    net = TinySSD(num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iter, _ = d2l.load_data_bananas(batch_size=32)
    trainer = Trainer(net, train_iter, device)
    trainer.train(num_epochs=20)
