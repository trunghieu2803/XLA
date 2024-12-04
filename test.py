import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading


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

        # Nút chọn hình ảnh
        self.btn_select_image = tk.Button(root, text="Chọn Hình Ảnh", command=self.select_image)
        self.btn_select_image.pack()

        # Nút bật video webcam
        self.btn_start_video = tk.Button(root, text="Bật Video Webcam", command=self.start_video)
        self.btn_start_video.pack()

        # Nút chọn video từ máy tính
        self.btn_select_video = tk.Button(root, text="Chọn Video", command=self.select_video)
        self.btn_select_video.pack()

        # Khung hiển thị ảnh gốc
        self.frame_original = tk.LabelFrame(root, text="Ảnh Gốc", width=500, height=500)
        self.frame_original.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas_original = tk.Canvas(self.frame_original, width=500, height=500)
        self.canvas_original.pack()

        # Khung hiển thị ảnh/video đã xử lý
        self.frame_processed = tk.LabelFrame(root, text="Ảnh/Video Sau Xử Lý", width=640, height=480)
        self.frame_processed.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas_processed = tk.Canvas(self.frame_processed, width=640, height=480)
        self.canvas_processed.pack()

        # Xử lý sự kiện đóng ứng dụng
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def select_image(self):
        # if self.cap is not None:
        #     self.on_closing()
        self.img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
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


    def start_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Không thể mở camera.")
                return

        self.running = True
        threading.Thread(target=self.process_video).start()

    def process_video(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model.predict(frame)
            for r in results:
                img_arr = r.plot()
            img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            self.canvas_processed.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas_processed.image = img

        if self.cap is not None:
            self.cap.release()

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"Error: Không thể mở video từ {self.video_path}")
                return

            self.running = True
            threading.Thread(target=self.process_selected_video).start()

    def process_selected_video(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model.predict(frame)
            for r in results:
                img_arr = r.plot()
            img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            self.canvas_processed.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas_processed.image = img

        if self.cap is not None:
            self.cap.release()

    def on_closing(self):
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
