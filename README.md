# vit-ssd

- base code 
- vit : https://github.com/YoojLee/vit
- ssd : https://d2l.ai/chapter_computer-vision/ssd.html
- 2개의 모델 구현(banana and apple classification, idol face classification)

# Data set
1. D2L bananas data set에 apple 추가한 커스텀 데이터 (train : 1000, test : 100)
2. 아이돌 이미지 COCO data set, 커스텀 데이터 (train : 1548, validation : 80, test : 80)
   (Robofolw활용)

# SSD Reimplementation
- [ ] SSD.py

출처 -- https://d2l.ai/chapter_computer-vision/ssd.html

- D2L SSD(single shot detector)의 tinySSD 코드 활용


# ViT Reimplementation
- [ ] model.py
- [ ] train.py
- [ ] loss.py
- [ ] dataset.py

출처 -- https://daebaq27.tistory.com/112



- augmentation.py: 간단한 Dataset Transform을 구현
- dataset.py: ImageNet-1k dataset 클래스 구현
- model.py: ViT 모델 구현
- scheduler.py: linear warm-up + cosine annealing 등의 스케쥴러 구현.
- train.py: single gpu 상황을 가정한 train.py
- train_multi.py: multi-gpu 하에서의 train.py
- utils.py: metrics 계산, checkpoint load 등의 여러 함수 구현
