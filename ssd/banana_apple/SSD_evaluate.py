import torch
from torch import nn
import d2l

from torch.utils.data import Dataset
from PIL import Image
import os

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.ops import nms
import torch.nn.functional as F
from torchvision.ops import box_iou

import json

from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from SSD import TinySSD, CustomImageDataset

class Evaluator():
    def __init__(self, net, data_loader, device):
        self.net = net
        self.data_loader = data_loader
        self.device = device

    def multibox_detection(self, cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
        """Predict bounding boxes using non-maximum suppression.

        Defined in :numref:`subsec_predicting-bounding-boxes-nms`"""
        batch_size = cls_probs.shape[0]
        anchors = anchors.squeeze(0)
        num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
        out = []
        for i in range(batch_size):
            cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
            conf, class_id = torch.max(cls_prob[1:], 0)
            predicted_bb = self.offset_inverse(anchors, offset_pred)
            predicted_bb = torch.from_numpy(predicted_bb)
            keep = nms(predicted_bb, conf, nms_threshold)
            # Find all non-`keep` indices and set the class to background
            all_idx = torch.arange(num_anchors, dtype=torch.long, device='cpu')
            combined = torch.cat((keep, all_idx))
            uniques, counts = combined.unique(return_counts=True)
            non_keep = uniques[counts == 1]
            all_id_sorted = torch.cat((keep, non_keep))
            class_id[non_keep] = -1
            class_id = class_id[all_id_sorted]
            conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
            # Here `pos_threshold` is a threshold for positive (non-background)
            # predictions
            below_min_idx = (conf < pos_threshold)
            class_id[below_min_idx] = -1
            conf[below_min_idx] = 1 - conf[below_min_idx]
            pred_info = torch.cat((class_id.unsqueeze(1),
                                   conf.unsqueeze(1),
                                   predicted_bb), dim=1)
            out.append(pred_info)
        return torch.stack(out)


    def box_corner_to_center(self,boxes):
        """Convert from (upper_left, bottom_right) to (center, width, height)"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        boxes = torch.stack((cx, cy, w, h), axis=-1)
        return boxes

    def offset_inverse(self, anchors, offset_preds):
        """Predict bounding boxes based on anchor boxes with predicted offsets.

        Defined in :numref:`subsec_labeling-anchor-boxes`"""
        anc = self.box_corner_to_center(anchors)
        pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
        pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
        pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)
        xmin_ymin = pred_bbox[:, :2] - pred_bbox[:, 2:] / 2
        xmax_ymax = pred_bbox[:, :2] + pred_bbox[:, 2:] / 2
        predicted_bbox = np.hstack((xmin_ymin, xmax_ymax))

        return predicted_bbox

    def evaluate(self):
        self.net = self.net.to(self.device)
        self.net.eval()
        idx_img = 0
        threshold = 0.5
        with torch.no_grad():
            for images, targets in self.data_loader:
                images = images.to(self.device)

                boxes = []
                label = []
                for target in targets:
                    boxes.append(target['boxes'].to(self.device))
                    label.append(target['image_id'].to(self.device))

                anchors, cls_preds, bbox_preds = self.net(images)  # 예측 값 가져오기
                cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
                output = self.multibox_detection(cls_probs, bbox_preds, anchors)  # 실제 예측된 것들 중에 신뢰도 높은 박스 선택 하는 함수
                # (1,6) : [class id, score, x, y, w, h]
                # class id : 배경(-1) or class (0~)

                idx = [i for i, row in enumerate(output[0]) if row[0] != -1]  # -1 (배경 클래스) 이 아닌 경우 선택
                output = output[0, idx]  # 최종 선택 된 박스 위치와 클래스 전달 (1,6) 짜리가 선택된 박스만큼 존재

                # NMS 적용
                # output에서 class id, score, x, y, w, h를 분리
                class_ids = output[:, 0]
                scores = output[:, 1]
                boxes = output[:, 2:]

                # NMS 적용 (iou 임계값은 0.5로 설정)
                # IoU 값은 0에서 1 사이의 값으로, 두 박스가 완전히 겹칠 경우 1, 전혀 겹치지 않을 경우 0의 값을 가짐 (20% 이상 겹치면 같은 물체로 간주)
                keep = nms(boxes, scores, iou_threshold=0.3)

                # NMS에 의해 선택된 박스만 리턴
                result = output[keep]

                # 이미지를 numpy 배열로 변환
                image_np = images.cpu().numpy()[0].transpose((1, 2, 0))

                # 이미지의 크기를 가져옴
                height, width = 360, 360

                fig, ax = plt.subplots(1)
                ax.imshow(image_np)

                for i in range(result.shape[0]):
                    label, score, x, y, w, h = result[i]
                    if score < threshold:
                        continue
                    # 비율을 실제 픽셀 값으로 변환
                    x *= width
                    y *= height
                    w *= width
                    h *= height

                    # Rectangle patch 생성
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

                    # Rectangle patch를 이미지에 추가
                    ax.add_patch(rect)

                    # 클래스 ID와 스코어를 이미지에 추가
                    plt.text(x, y, f'{label.item()}: {score.item()}', color='white', fontsize=10,
                             bbox=dict(facecolor='red', alpha=0.5))

                # 이미지를 파일로 저장
                plt.savefig(f'predicted_images/output_image_{idx_img}.png')
                idx_img += 1
                print(idx_img)

def custom_collate_fn(batch):
    # Separate the images and the targets in the batch
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # You can stack the images because they have the same size
    images = default_collate(images)

    return images, targets

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((360, 360)),  # 이미지 크기 조절
        transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환
    ])
    model = "best_model.pth"

    # 데이터 로드
    print("데이터 로드중...")
    test_dataset = CustomImageDataset('/Users/apple/PycharmProjects/vit-ssd/ssd/face.v9i.coco/test/_annotations.coco.json', '/Users/apple/PycharmProjects/vit-ssd/ssd/face.v9i.coco/test', transform=transform)
    test_iter = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate_fn)

    # 모델 로드
    net = TinySSD(num_classes=1)
    net.load_state_dict(torch.load(model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 평가
    evaluator = Evaluator(net, test_iter, device)
    evaluator.evaluate()
