from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()


def download_data(version=4):
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API"))
    project = rf.workspace("kartiks-workspace-ia4hy").project(
        "basketball-players-arj24"
    )
    version = project.version(4)
    dataset = version.download("yolov8")


if __name__ == "__main__":
    download_data()
