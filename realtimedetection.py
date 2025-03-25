import cv2
from keras.models import model_from_json
import numpy as np
import os

# ======= TẢI MÔ HÌNH =======
json_file = open("model/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model/facialemotionmodel.h5")

# ======= LOAD HAAR CASCADE =======
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# ======= HÀM TIỀN XỬ LÝ ẢNH =======
def extract_features(image):
    feature = np.array(image, dtype=np.float32)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0  # Chuẩn hóa

# ======= GÁN NHÃN CẢM XÚC =======
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# ======= THƯ MỤC =======
input_folder = 'test'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# ======= ĐỌC DANH SÁCH ẢNH =======
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Đang xử lý {len(image_files)} ảnh trong thư mục '{input_folder}'...")

# ======= XỬ LÝ TỪNG ẢNH =======
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Không đọc được ảnh: {image_file}")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    detected_emotions = []

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)

        pred = model.predict(img)[0]
        predicted_label = labels[np.argmax(pred)]
        confidence = np.max(pred)

        detected_emotions.append(predicted_label)

        # Vẽ khung + nhãn từng khuôn mặt trên ảnh gốc
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)  # Xanh nếu độ tin cậy >50%, đỏ nếu thấp hơn
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{predicted_label} ({confidence:.2%})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Nếu không phát hiện khuôn mặt thì gán "unknown"
    if not detected_emotions:
        detected_emotions.append("unknown")

    # Chọn nhãn phổ biến nhất nếu có nhiều khuôn mặt
    final_emotion = max(set(detected_emotions), key=detected_emotions.count)

    # Ghi nhãn tổng kết lên ảnh
    text = f"Emotion: {final_emotion}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Lưu ảnh đã nhận diện vào thư mục output
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, frame)

print("Hoàn tất xử lý, kết quả đã lưu vào thư mục 'output'.")
