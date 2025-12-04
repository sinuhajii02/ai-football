from ultralytics import YOLO

model = YOLO('models/best.pt')

train_result = model.predict('input_videos/B1606b0e6_1 (26).mp4', save=True)
print(train_result[0])
print("================================================")
for box in train_result[0].boxes:
    print(box)