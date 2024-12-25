import numpy as np
import cv2
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Thiết lập thông số
color = 0
smoke = 0
inten = 0
res = 0
def enlarge_box(x, y, w, h, scale=1.2):
    center_x, center_y = x + w // 2, y + h // 2
    new_w, new_h = int(w * scale), int(h * scale)
    new_x, new_y = center_x - new_w // 2, center_y - new_h // 2
    return new_x, new_y, new_w, new_h

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Ngưỡng xác định ngọn lửa và chuyển động
threshold_area = 200  # Tăng diện tích tối thiểu cho vùng phát hiện lửa để giảm độ nhạy

# Khởi tạo mô hình phát hiện khuôn mặt và phần thân trên
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Hàm Non-Maximum Suppression (NMS) để loại bỏ các phát hiện chồng chéo
def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Hàm kiểm tra hình dạng contour (phát hiện lửa)
def is_flame_shape(cnt):
    area = cv2.contourArea(cnt)
    if area < threshold_area:
        return False
    perimeter = cv2.arcLength(cnt, True)
    shape_factor = 4 * np.pi * (area / (perimeter ** 2))
    return 0.2 < shape_factor < 0.5  # Điều kiện hệ số hình dạng chặt hơn để tránh các phát hiện sai

# Vòng lặp xử lý video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc từ camera.")
        break

    # Chuyển sang không gian màu HSV và làm mờ Gaussian để giảm nhiễu
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Ngưỡng màu ngọn lửa trong không gian màu HSV, giới hạn chặt chẽ hơn
    lower_bound = np.array([25, 100, 100], dtype="uint8")  # Màu cam đỏ sáng
    upper_bound = np.array([30, 255, 255], dtype="uint8")
    mask = cv2.inRange(blurred, lower_bound, upper_bound)

    # Tìm đường viền và xác định vùng ngọn lửa
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flame_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < threshold_area:
            continue
        if is_flame_shape(cnt):  # Chỉ giữ lại những contour có hình dạng đặc trưng của ngọn lửa
            x, y, w, h = cv2.boundingRect(cnt)
            flame_boxes.append([x, y, x + w, y + h])

    # Áp dụng NMS để lọc phát hiện trùng lặp
    flame_boxes = non_max_suppression_fast(np.array(flame_boxes), 0.5)
    for box in flame_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, 'Flame', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        color = 1

    # Phát hiện chuyển động sử dụng khung xám và bộ phát hiện bóng
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 5)
    fmask = fgbg.apply(gray_blurred)
    kernel = np.ones((20, 20), np.uint8)
    fmask = cv2.dilate(fmask, kernel)

    # Ngưỡng nhị phân và lọc nhiễu
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Tìm đường viền và kiểm tra chuyển động mạnh (intensity)
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inten = 1 if contours else 0

    # Phát hiện người trong khung hình
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20))
    upperbodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    # Vẽ hình chữ nhật xung quanh các khuôn mặt phát hiện được
    for (x, y, w, h) in faces:
        x, y, w, h = enlarge_box(x, y, w, h, scale=1.5)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Vẽ hình chữ nhật xung quanh phần thân trên phát hiện được
    for (x, y, w, h) in upperbodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, 'Upper Body', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Gửi email nếu phát hiện lửa (Tạm thời comment phần này)
    # if color == 1 and res == 0:
    #     res = 1
    #     send_email()
    # elif color == 0:
    #     res = 0

    cv2.imshow('Processed Frame', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break
    

cap.release()
cv2.destroyAllWindows()

