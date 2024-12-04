from ultralytics import YOLO
import torch

# Clear GPU cache to ensure optimal usage
torch.cuda.empty_cache()


def main():
    # Initialize YOLO model with a pre-trained weight file
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="data.yaml",  # Path to the dataset configuration
        epochs=50,  # Number of epochs to train
        imgsz=640,  # Image size
        device='cuda',  # Device to use for training (GPU)
        workers=4,  # Number of data loading workers
        batch=64  # Batch size
    )


if __name__ == '__main__':
    main()
