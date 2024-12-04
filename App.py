import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import time


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and Video Face Detection")

        # Đường dẫn mô hình YOLO
        self.model_path = "D:/JetBrains/PyCharm/Source/AgeGenderDetection/runs/detect/train3/weights/best.pt"
        self.model = YOLO(self.model_path)

        # Đường dẫn ảnh/video
        self.img_path = ""
        self.video_path = ""
        self.cap = None  # VideoCapture object
        self.running = False  # Trạng thái video

        # Tạo thư mục lưu ảnh nếu chưa tồn tại
        self.save_dir = "saved_images"
        os.makedirs(self.save_dir, exist_ok=True)

        # Tạo giao diện
        self.create_widgets()

        # Xử lý sự kiện đóng ứng dụng
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Khung chứa nút
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=0, column=0, columnspan=2, pady=10)

        # Nút chọn hình ảnh
        self.btn_select_image = tk.Button(button_frame, text="Chọn Hình Ảnh", command=self.select_image, width=20)
        self.btn_select_image.grid(row=0, column=0, padx=5)

        # Nút bật video webcam
        self.btn_start_video = tk.Button(button_frame, text="Bật Video Webcam", command=self.start_video, width=20)
        self.btn_start_video.grid(row=0, column=1, padx=5)

        # Nút chọn video từ máy tính
        self.btn_select_video = tk.Button(button_frame, text="Chọn Video", command=self.select_video, width=20)
        self.btn_select_video.grid(row=0, column=2, padx=5)

        # Nút lưu ảnh đã xử lý
        self.btn_save_image = tk.Button(button_frame, text="Lưu Ảnh Đã Xử Lý", command=self.save_processed_image, width=20)
        self.btn_save_image.grid(row=0, column=3, padx=5)

        # Khung hiển thị ảnh gốc
        self.frame_original = tk.LabelFrame(self.root, text="Ảnh Gốc", width=500, height=500)
        self.frame_original.grid(row=1, column=0, padx=10, pady=10)
        self.canvas_original = tk.Canvas(self.frame_original, width=500, height=500)
        self.canvas_original.pack()

        # Khung hiển thị ảnh/video đã xử lý
        self.frame_processed = tk.LabelFrame(self.root, text="Ảnh/Video Sau Xử Lý", width=640, height=480)
        self.frame_processed.grid(row=1, column=1, padx=10, pady=10)
        self.canvas_processed = tk.Canvas(self.frame_processed, width=640, height=480)
        self.canvas_processed.pack()

    def select_image(self):
        self.cancel_video()  # Hủy video nếu đang chạy
        self.img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

        # Xóa nội dung canvas gốc và canvas đã xử lý
        self.canvas_original.delete("all")
        self.canvas_processed.delete("all")

        if self.img_path:
            image = Image.open(self.img_path).resize((500, 500))
            self.photo_original = ImageTk.PhotoImage(image)
            self.canvas_original.create_image(0, 0, anchor=tk.NW, image=self.photo_original)
            self.process_image()

    def process_image(self):
        if self.img_path:
            img = cv2.imread(self.img_path)
            if img is not None:
                results = self.model.predict(img)
                for r in results:
                    img_arr = r.plot()
                    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                    processed_image = Image.fromarray(img_rgb).resize((500, 500))
                    self.photo_processed = ImageTk.PhotoImage(processed_image)
                    self.canvas_processed.create_image(0, 0, anchor=tk.NW, image=self.photo_processed)

                    # # Tự động lưu ảnh sau khi phân tích
                    # save_path = os.path.join(self.save_dir, "processed_image.jpg")
                    # cv2.imwrite(save_path, img_arr)
                    # print(f"Ảnh đã được lưu tại: {save_path}")

    def save_processed_image(self):
        """Lưu ảnh đã xử lý vào thư mục."""
        if hasattr(self, 'photo_processed') and self.photo_processed is not None:
            save_path = os.path.join(self.save_dir, "processed_image_manual.jpg")
            img = cv2.imread(self.img_path)
            results = self.model.predict(img)
            for r in results:
                img_arr = r.plot()
                cv2.imwrite(save_path, img_arr)
                print(f"Ảnh đã được lưu thủ công tại: {save_path}")

    def start_video(self):
        self.cancel_video()  # Hủy video nếu đang chạy
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Không thể mở camera.")
            return

        self.running = True
        threading.Thread(target=self.process_video).start()

    def process_video(self):
        frame_count = 0
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model.predict(frame)
            for r in results:
                img_arr = r.plot()

            # Lưu khung hình mỗi giây
            if frame_count % 15 == 0:  # Tương đương 0.5 giây ở tốc độ 30 fps
                save_path = os.path.join(self.save_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(save_path, img_arr)

            img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            self.canvas_processed.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas_processed.image = img

            frame_count += 1
            time.sleep(0.5)  # Cập nhật trạng thái video sau mỗi 0.5 giây

        if self.cap is not None:
            self.cap.release()

    def select_video(self):
        self.cancel_video()  # Hủy video nếu đang chạy
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"Error: Không thể mở video từ {self.video_path}")
                return

            self.running = True
            threading.Thread(target=self.process_selected_video).start()

    def process_selected_video(self):
        frame_count = 0
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model.predict(frame)
            for r in results:
                img_arr = r.plot()

            # Lưu khung hình mỗi giây
            if frame_count % 15 == 0:  # Tương đương 0.5 giây ở tốc độ 30 fps
                save_path = os.path.join(self.save_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(save_path, img_arr)

            img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            self.canvas_processed.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas_processed.image = img

            frame_count += 1
            time.sleep(0.5)

        if self.cap is not None:
            self.cap.release()

    def cancel_video(self):
        """Hủy phát video hoặc dừng webcam."""
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.canvas_processed.delete("all")  # Xóa nội dung canvas

    def on_closing(self):
        self.cancel_video()  # Hủy video nếu đang chạy
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
