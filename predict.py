from ultralytics import YOLO
import cv2
from PIL import Image

# Đường dẫn đến mô hình đã huấn luyện
model_path = "D:/JetBrains/PyCharm/Source/AgeGenderDetection/runs/detect/train3/weights/best.pt"

if __name__ == '__main__':
    # Tải mô hình YOLO
    model = YOLO(model_path)

    # Mở camera
    cap = cv2.VideoCapture(0)  # 0 là chỉ số của camera mặc định; nếu dùng camera khác, thay số khác

    if not cap.isOpened():
        print("Error: Không thể mở camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Không thể đọc frame từ camera.")
            break

        # Dự đoán trên frame
        results = model.predict(frame)

        # Hiển thị kết quả trên frame
        for r in results:
            # Vẽ kết quả lên frame
            img_arr = r.plot()

            # Chuyển đổi từ BGR sang RGB
            img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

            # Hiển thị video
            cv2.imshow("Age and Gender Detection", img_arr)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
