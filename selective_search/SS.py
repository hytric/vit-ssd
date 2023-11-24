import selectivesearch
import cv2
import matplotlib.pyplot as plt
import os


img=cv2.imread('newjeans.jpeg')
img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('img shape:', img.shape)

plt.figure(figsize=(8,8))
plt.imshow(img_rgb)
plt.show()

_, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=1000)

cand_rects = [cand['rect'] for cand in regions]
green_rgb = (125, 255, 51) # bounding box color
img_rgb_copy = img_rgb.copy() # 이미지 복사

for rect in cand_rects:
  left = rect[0]
  top = rect[1]
  # rect[2], rect[3]은 너비와 높이이므로 우하단 좌표를 구하기 위해 좌상단 좌표에 각각을 더함.
  right = left + rect[2]
  bottom = top + rect[3]
  img_rgb_copy = cv2.rectangle(img_rgb_copy, (left, top), (right, bottom), color=green_rgb, thickness=2)

plt.figure(figsize=(8, 8))
plt.imshow(img_rgb_copy)
plt.show()