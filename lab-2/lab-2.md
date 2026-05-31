```
%pip install -q ultralytics roboflow
```

```
import os
import cv2
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
from ultralytics import YOLO
```

```
os.chdir('/kaggle/working')

rf = Roboflow(api_key="Q25aGlkLeZtAik9jhrtr")

project_toronto = rf.workspace("university-of-toronto-xho85").project("numberdetection-eppfj")
dataset_toronto = project_toronto.version(2).download("yolov8")

project_svhn = rf.workspace("peking-uni").project("svhn-rktm0")

for v in [1, 2, 3, 4]:
    try:
        dataset_svhn = project_svhn.version(v).download("yolov8")
        break
    except Exception:
        pass
```

```
import yaml

with open(f"{dataset_svhn.location}/data.yaml", 'r') as f:
    data = yaml.safe_load(f)

img_path = glob.glob(f"{dataset_svhn.location}/train/images/*.jpg")[0]
label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

with open(label_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    class_id, x_center, y_center, width, height = map(float, line.strip().split())
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h)
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, str(int(class_id)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

```
model = YOLO('yolov8m.pt')

results_pretrain = model.train(
    data=f"{dataset_toronto.location}/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    name='yolov8_toronto_pretrain',
    device='cuda:0',
    verbose=False
)
```

```
pretrained_weights = '/kaggle/working/runs/detect/yolov8_toronto_pretrain/weights/best.pt'
model_finetune = YOLO(pretrained_weights)

results_finetune = model_finetune.train(
    data=f"{dataset_svhn.location}/data.yaml",
    epochs=10,
    imgsz=480,
    batch=16,
    name='yolov8_svhn_finetune',
    device='cuda:0',
    verbose=False
)
```

```
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

best_model = YOLO('/kaggle/working/runs/detect/yolov8_svhn_finetune/weights/best.pt')
metrics = best_model.val(data=f"{dataset_svhn.location}/data.yaml", split='test', verbose=False)

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall: {metrics.box.r.mean():.4f}")

val_images = glob.glob(f"{dataset_svhn.location}/valid/images/*.jpg")
total_iou = 0
total_boxes = 0

for img_p in val_images[:100]: 
    lbl_p = img_p.replace('images', 'labels').replace('.jpg', '.txt')
    if not os.path.exists(lbl_p): continue
    
    img = cv2.imread(img_p)
    h, w, _ = img.shape
    
    with open(lbl_p, 'r') as f:
        gt_lines = f.readlines()
        
    gt_boxes = []
    for line in gt_lines:
        _, xc, yc, bw, bh = map(float, line.strip().split())
        gt_boxes.append([
            (xc - bw/2)*w, (yc - bh/2)*h, 
            (xc + bw/2)*w, (yc + bh/2)*h
        ])
        
    preds = best_model(img_p, verbose=False)[0]
    pred_boxes = preds.boxes.xyxy.cpu().numpy()
    
    for pd_box in pred_boxes:
        best_iou = 0
        for gt_box in gt_boxes:
            iou = calculate_iou(pd_box, gt_box)
            if iou > best_iou:
                best_iou = iou
        total_iou += best_iou
        total_boxes += 1

mean_iou = total_iou / total_boxes if total_boxes > 0 else 0
print(f"Custom Calculated Mean IoU (Validation Subset): {mean_iou:.4f}")
```

```
my_photos_dir = '/kaggle/input/datasets/justneedcoffee/license-plates/'
if not os.path.exists(my_photos_dir):
    search_res = glob.glob('/kaggle/input/**/license-plates', recursive=True)
    if search_res: my_photos_dir = search_res[0]

output_dir = '/kaggle/working/inference_results'
os.makedirs(output_dir, exist_ok=True)

test_photos = glob.glob(f"{my_photos_dir}/*.jpg") + glob.glob(f"{my_photos_dir}/*.png")

for photo_path in test_photos:
    results = best_model.predict(
        source=photo_path, 
        conf=0.15, 
        iou=0.4, 
        imgsz=1280, 
        save=True, 
        project=output_dir, 
        name='predicts', 
        exist_ok=True,
        verbose=False
    )

res_images = glob.glob(f"{output_dir}/predicts/*.jpg")
for img_path in res_images[:5]:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
```