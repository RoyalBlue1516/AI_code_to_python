import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# 영상 경로와 출력 프레임 저장 경로 설정
video_path = './input_video.mp4'
output_frame_path = './frames/'  # 추출된 프레임 저장 경로
augmented_frame_path = './augmented_frames/'  # 증강된 프레임 저장 경로

# 디렉토리 생성
os.makedirs(output_frame_path, exist_ok=True)
os.makedirs(augmented_frame_path, exist_ok=True)

#클래스ID생성(예시)
classes = {
    'person': 0,
    'car': 1,
    'dog': 2
}

# 바운딩 박스와 클래스 레이블 예시 (프레임 수에 맞게 각 프레임에 할당)
# 실제 바운딩 박스와 레이블이 있어야 함
example_bboxes = [
    [[0.5, 0.5, 0.2, 0.2]]  # 각 프레임에 대한 bbox 좌표 (x_center, y_center, width, height)

]
example_labels = [
    [0]  # 클래스 ID
   
]

# 데이터 증강을 위한 Albumentations 변환 정의
def augment_frame_with_bboxes(frame, bboxes, class_labels):

    # 항상 적용할 기본 변환
    base_transform = A.Compose([
        A.Resize(416, 416),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # 개별로 적용할 변환 정의
    individual_transforms = {
        "rotate": A.RandomRotate90(p=1),
        "flip_hor": A.HorizontalFlip(p=0.5),
        "flip_ver": A.VerticalFlip(p=0.5),
        "transpose": A.Transpose(),
        "brightness_contrast": A.RandomBrightnessContrast(p=1.0),  # 항상 적용 p=1.0 
        "blur": A.Blur(blur_limit=3)
    }

    # 각 변환을 개별적으로 적용하여 이미지와 바운딩 박스 반환
    for name, ind_transform in individual_transforms.items():
        # 기본 변환 + 개별 변환
        composed = A.Compose([
            ind_transform,               # 개별 변환 적용
            A.Resize(416, 416),     
            A.Normalize(),           
            ToTensorV2()             
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        # 변환 적용
        transformed = composed(image=frame, bboxes=bboxes, class_labels=class_labels)
    
    return transformed['image'], transformed['bboxes'], transformed['class_labels']
    

# 영상 데이터를 프레임으로 추출하는 함수
def extract_frames(video_path, output_frame_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frame_count = 0
    
    if not cap.isOpened():
        print(f"Error: 비디오를 열 수 없습니다: {video_path}")
        return

    # 영상에서 프레임을 추출
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # 더 이상 읽을 프레임이 없는 경우 종료

        # 일정 간격으로 프레임 저장
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{extracted_frame_count:05d}.jpg"
            frame_path = os.path.join(output_frame_path, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"{extracted_frame_count}개의 프레임을 추출했습니다.")

# 추출된 프레임을 증강하는 함수
def augment_frames(input_frame_path, output_frame_path, bboxes, class_labels):
    frame_files = [f for f in os.listdir(input_frame_path) if f.endswith('.jpg')]

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(input_frame_path, frame_file)
        
        # 프레임 읽기
        frame = cv2.imread(frame_path)

        # 바운딩 박스 정보 가져오기 (예시)
        # bboxes와 class_labels는 사전에 정의된 리스트
        # 각 프레임에 대한 bbox와 class label이 할당되어 있어야 함.
    # 바운딩 박스 정보 가져오기 (리스트가 부족할 경우 대비)
        if i < len(bboxes):
            frame_bboxes = bboxes[i]
            frame_labels = class_labels[i]
        else:
            # 바운딩 박스와 레이블 정보가 부족할 경우 마지막 값을 반복 사용
            frame_bboxes = bboxes[-1]
            frame_labels = class_labels[-1]

        # 증강 적용
        augmented_frame, aug_bboxes, aug_labels = augment_frame_with_bboxes(frame, frame_bboxes, frame_labels)

        # 증강된 이미지 저장
        augmented_frame_filename = f"aug_{frame_file}"
        augmented_frame_path = os.path.join(output_frame_path, augmented_frame_filename)
        augmented_frame_np = augmented_frame.permute(1, 2, 0).cpu().numpy()  # 텐서를 이미지로 변환
        cv2.imwrite(augmented_frame_path, augmented_frame_np)

        # YOLO 레이블 파일 저장
        label_filename = augmented_frame_filename.replace('.jpg', '.txt')
        label_path = os.path.join(output_frame_path, label_filename)
        with open(label_path, 'w') as f:
            for bbox, label in zip(aug_bboxes, aug_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{label} {x_center} {y_center} {width} {height}\n")

    print(f"{len(frame_files)}개의 프레임을 증강")

# 프레임 추출 
extract_frames(video_path, output_frame_path)

# 증강 수행
augment_frames(output_frame_path, augmented_frame_path, example_bboxes, example_labels)

