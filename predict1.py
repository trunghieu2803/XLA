from ultralytics import YOLO
import cv2
from PIL import Image

model_path = "D:/JetBrains/PyCharm/Source/AgeGenderDetection/runs/detect/train3/weights/best.pt"
img_path = "D:/JetBrains/PyCharm/Source/AgeGenderDetection/test5.jpg"

if __name__ == '__main__':
    # Load the YOLO model
    model = YOLO(model_path)

    # Read the image
    img = cv2.imread(img_path)

    if img is not None:
        # Perform prediction
        results = model.predict(img)

        # Process and display results
        for r in results:
            # Plot the results
            img_arr = r.plot()

            # Convert from BGR to RGB
            im = Image.fromarray(img_arr[..., ::-1])
            im.show()
    else:
        print(f"Error: Could not read image from {img_path}")