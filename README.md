# youtube-video-shorts-generator
An automated tool to generate short vertical videos from YouTube content, focusing on the most replayed sections and using AI-powered auto-reframing.
# YouTube Shorts Generator

This tool automatically generates short, vertical videos from YouTube content, focusing on the most replayed sections and using AI-powered auto-reframing.

## Features

- Downloads YouTube videos
- Identifies most replayed sections
- Generates short clips from these sections
- Auto-reframes videos to vertical format using AI
- Supports multi-processing for faster generation
- Resumes interrupted processing

## Prerequisites

- Python 3.8+
- FFmpeg with GPU support (for NVIDIA GPUs)
- CUDA-compatible GPU (for optimal performance)

## Installation

1. Clone the repository:
git clone https://github.com/Enigmaticelectro/youtube-video-shorts-generator.git
cd youtube-shorts-generator
Copy
2. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Copy
3. Install the required packages:
pip install -r requirements.txt
Copy
4. Download the YOLOv8 model:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
Copy
## Configuration

Create a `config.ini` file in the project root with the following content:

```ini
[DEFAULT]
model_path = yolov8m.pt
video_quality = bestvideo[height>=1080]+bestaudio/best[height>=1080]
peak_prominence = 0.1
output_dir = shorts
num_workers = 4
short_duration = 45
proxy_scale = 0.3
skip_frames = 3
Adjust these values as needed for your setup.
```
Usage
Run the script with a YouTube URL:
Copypython main.py --youtube_url https://www.youtube.com/watch?v=VIDEO_ID
Additional options:
```
--config: Specify a custom config file path
--model_path: Path to the YOLO model file
--video_quality: YouTube video download quality
--peak_prominence: Peak detection sensitivity for identifying popular segments
--output_dir: Output directory for generated shorts
--num_workers: Number of worker processes for parallel processing
--short_duration: Target duration of each short in seconds
--proxy_scale: Scale factor for proxy video used in processing
--skip_frames: Number of frames to skip during processing
```
Output
Generated short videos will be saved in the specified output directory, organized by video ID.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Copy
To set up the script:
```
1. Create a new repository on GitHub with the name "youtube-shorts-generator".
2. Initialize a git repository in your local project folder.
3. Add all your script files, including `main.py`, `requirements.txt`, and the `README.md`.
4. Commit these files and push them to your GitHub repository.
5. Create a `requirements.txt` file listing all the Python packages required for your script. You can generate this using `pip freeze > requirements.txt` in your virtual environment.
6. Create a `LICENSE` file if you want to specify a license for your project.
```
This setup will provide a clear guide for users on how to install, configure, and use your YouTube Shorts Generator script.
