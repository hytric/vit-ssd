from vit import ViT
from torchvision.transforms import functional as F
from torchvision.ops import nms

class SSD_ViT(nn.Module):
    def __init__(self, ssd_model, vit_model):
        super(SSD_ViT, self).__init__()
        self.ssd = ssd_model
        self.vit = vit_model

    def forward(self, x):
        batch_size = x.size(0)

        # SSD 모델을 통해 객체 탐지
        anchors, cls_preds, bbox_preds = self.ssd(x)

        # 가장 높은 확률을 가진 bounding box 선택
        max_boxes = []
        for i in range(batch_size):
            scores, labels = cls_preds[i].max(1)
            max_score_idx = scores.argmax(0).item()
            max_box = anchors[i][max_score_idx]
            max_boxes.append(max_box)

        # 선택된 bounding box를 사용하여 이미지에서 객체를 잘라냄
        outputs = []
        for i in range(batch_size):
            bbox = max_boxes[i].tolist()
            cropped_image = F.crop(x[i], bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])

            # 이미지를 240*240 크기로 재조정
            cropped_image = F.resize(cropped_image, [240, 240])

            # 잘라낸 이미지를 ViT 모델에 입력
            out = self.vit(cropped_image.unsqueeze(0))  # 배치 차원 추가
            outputs.append(out)

        return torch.cat(outputs, dim=0)


# SSD 모델 로드
ssd_net = TinySSD(num_classes=1)
ssd_net.load_state_dict(torch.load('model_final.pth'))

# ViT 모델 로드
vit_net = ViT(image_size=240, patch_size=16, num_classes=1000, dim=1024, depth=6, heads=16, mlp_dim=2048)

# SSD-ViT 모델 생성
model = SSD_ViT(ssd_net, vit_net)
