import torch
from ultralytics import YOLO
import traceback


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

        results = model.train(
            data="data/data.yaml",
            epochs=80,
            imgsz=832,
            batch=16,
            device=device,
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,
            workers=4,
            project="runs",
            name="basketball_model",
            exist_ok=True,  # 🔥 prevents crash if folder exists
        )

        print("\n✅ Training Completed Successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (Ctrl+C). Saving progress...")

    except Exception as e:
        print("\n❌ Training crashed!")
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        print("\n📦 Check 'runs/' folder for saved weights (best.pt / last.pt)\n")


if __name__ == "__main__":
    main()
