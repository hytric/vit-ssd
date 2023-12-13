import cv2
import json
import os

# 이미지 파일이 저장된 디렉토리
img_dir = '/Users/apple/PycharmProjects/vit-ssd/ssd/dataset'  # 실제 이미지 디렉토리로 변경

# 저장할 JSON 파일명
json_file = 'labels.json'

# bounding box의 시작점, 끝점
start_point = None
end_point = None

# bounding box와 라벨을 저장할 딕셔너리
labels = {}

# 가능한 라벨 리스트
possible_labels = [0, 1, 2]  # 실제 사용할 라벨로 변경


def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # 왼쪽 마우스 버튼이 눌리면 시작점 지정
        start_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        # 왼쪽 마우스 버튼이 떼어지면 끝점 지정
        end_point = (x, y)
        # 사각형 그리기
        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow('image', img)

        # 라벨 입력 받기
        while True:
            label = input('Enter a label for this box(0 : 장원영, 1 : 안유진, 2 : 권은비) : ')
            if int(label) not in possible_labels:
                print('Invalid label. Please enter again.')
            else:
                break

        # 좌표와 라벨 저장
        if img_name not in labels:
            labels[img_name] = {'boxes': [], 'labels': []}

        # 좌표를 [0, 1] 범위로 정규화
        h, w = img.shape[:2]
        normalized_box = [start_point[0] / w, start_point[1] / h, end_point[0] / w, end_point[1] / h]

        labels[img_name]['boxes'].append(normalized_box)
        labels[img_name]['labels'].append(int(label))


for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    print(img_name)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) == 27:  # ESC 키
            break

    cv2.destroyAllWindows()

# JSON 파일로 저장
with open(json_file, 'w') as f:
    json.dump(labels, f)
