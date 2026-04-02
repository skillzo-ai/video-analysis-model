import torch
from ultralytics import YOLO
import traceback
import os


def get_device():
    if torch.cuda.is_available():
        device = 0
        name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        name = "CPU"
    return device, name


def main():

    device, device_name = get_device()
    print("\n" + "=" * 50)
    print(f"🚀 Training on: {device_name}")
    print("=" * 50 + "\n")

    try:
        model = YOLO("yolov8n.pt")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir = os.path.join(repo_root, "data")
        results = model.train(
            data="data/data.yaml",
            epochs=120,
            imgsz=832,
            batch=16,
            device=device,
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,
            workers=4,
            project=os.path.join(data_dir, "runs"),
            name="basketball_model",
            exist_ok=True,

            # 🔥 ADD THESE (HSV augmentation)
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,

            # 🔥 HIGHLY RECOMMENDED for your use case
            mosaic=1.0,
            mixup=0.1,

            # Optional but useful
            degrees=10,
            translate=0.1,
            scale=0.5
        )

        print("\n✅ Training Completed Successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (Ctrl+C). Saving progress...")

    except Exception as e:
        print("\n❌ Training crashed!")
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        print("\n📦 Check 'data/runs/' folder for saved weights (best.pt / last.pt)\n")


if __name__ == "__main__":
    main()
