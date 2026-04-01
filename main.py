import argparse
import os
from detection_pipeline.processor import VideoProcessor

def run_detection_pipeline(source: str, player_model_path: str = "best.pt", ball_model_path: str = "ball_detector_model.pt", output_path: str = "output_tracked.mp4"):
    """
    Function to be called from FastAPI or other modules.
    """
    if not os.path.exists(player_model_path):
        raise FileNotFoundError(f"Player model file not found: {player_model_path}")
    
    if not os.path.exists(ball_model_path):
        raise FileNotFoundError(f"Ball model file not found: {ball_model_path}")
    
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source video not found: {source}")

    processor = VideoProcessor(
        player_model_path=player_model_path, 
        ball_model_path=ball_model_path, 
        output_path=output_path
    )
    processor.process_video(source=source)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Basketball Detection and Tracking Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--player_model", type=str, default="best.pt", help="Path to player model weights")
    parser.add_argument("--ball_model", type=str, default="ball_detector_model.pt", help="Path to ball model weights")
    parser.add_argument("--output", type=str, default="output_tracked.mp4", help="Path to output video")
    
    args = parser.parse_args()

    try:
        run_detection_pipeline(
            source=args.source,
            player_model_path=args.player_model,
            ball_model_path=args.ball_model,
            output_path=args.output
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
