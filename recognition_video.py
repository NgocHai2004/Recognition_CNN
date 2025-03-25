import cv2
from keras.models import model_from_json
import numpy as np

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

# ======= MỞ CAMERA =======
cap = cv2.VideoCapture(0)  # 0 = webcam mặc định, đổi thành đường dẫn file nếu dùng video

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh sang grayscale để nhận diện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)

        pred = model.predict(img)[0]
        predicted_label = labels[np.argmax(pred)]
        confidence = np.max(pred)

        # Vẽ khung quanh khuôn mặt
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)  # Xanh nếu độ tin cậy cao, đỏ nếu thấp
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{predicted_label} ({confidence:.2%})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Hiển thị kết quả lên màn hình
    cv2.imshow("Facial Emotion Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
