import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT Engine for speed optimization.")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to input .pt model")
    parser.add_argument("--format", type=str, default="engine", choices=["engine", "onnx"], help="Format to export (engine/onnx)")
    parser.add_argument("--half", action="store_true", help="Use FP16 precision (highly recommended for GPUs)")
    parser.add_argument("--imgsz", type=int, default=832, help="Image size for the engine")
    parser.add_argument("--device", type=int, default=0, help="Device to compile on (usually 0 for main GPU)")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Exporting model to {args.format} with FP16={args.half} on imgsz={args.imgsz}")
    # Export the model
    # Note: TensorRT export requires `tensorrt` python package installed and NVIDIA CUDA/cuDNN configured properly.
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        dynamic=False  # static size is usually faster
    )
    
    print(f"Export successful. You can now pass the resulting {args.format} file to the processor script instead of the .pt file.")

if __name__ == "__main__":
    main()
