import os
import cv2
import subprocess
import requests
import yt_dlp
from scipy.signal import find_peaks
import logging
import configargparse
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up configuration
def parse_args():
    parser = configargparse.ArgParser(default_config_files=['config.ini'])
    parser.add_argument('-c', '--config', is_config_file=True, help='Config file path')
    parser.add_argument('--model_path', required=True, help='Path to the YOLO model file (e.g., yolov8m.pt)')
    parser.add_argument('--video_quality', default="bestvideo[height>=1080]+bestaudio/best[height>=1080]", help="Video download quality")
    parser.add_argument('--peak_prominence', type=float, default=0.1, help="Peak detection sensitivity")
    parser.add_argument('--youtube_url', required=True, help="YouTube URL to process")
    parser.add_argument('--output_dir', default="shorts", help="Output directory for short videos")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help="Number of worker processes")
    parser.add_argument('--short_duration', type=int, default=45, help="Target duration of each short in seconds")
    parser.add_argument('--proxy_scale', type=float, default=0.3, help="Scale factor for proxy video")
    parser.add_argument('--skip_frames', type=int, default=3, help="Number of frames to skip during processing")
    return parser.parse_args()

def extract_video_id(youtube_url):
    return youtube_url.split("=")[-1]

def get_most_replayed_parts(video_id, max_retries=5, backoff_factor=2):
    url = f"https://yt.lemnoslife.com/videos?part=mostReplayed&id={video_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    retries = 0
    delay = 2  # Initial delay

    while retries <= max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries <= max_retries:
                logger.warning(f"Error fetching data (attempt {retries}): {e}")
                logger.warning(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= backoff_factor  # Exponential backoff
            else:
                logger.error(f"Max retries exceeded. Error: {e}")
                return None

def find_top_replayed_timestamps(most_replayed_data, peak_prominence):
    if not most_replayed_data or "items" not in most_replayed_data or not most_replayed_data["items"]:
        logger.warning("No most replayed data available.")
        return []

    video_data = most_replayed_data["items"][0]
    if "mostReplayed" not in video_data or "markers" not in video_data["mostReplayed"]:
        logger.warning("Unexpected format in most replayed data.")
        return []

    markers = video_data["mostReplayed"]["markers"]
    intensities = [marker["intensityScoreNormalized"] for marker in markers]
    peaks, _ = find_peaks(intensities, prominence=peak_prominence)
    timestamps = [markers[i]["startMillis"] for i in peaks]
    return timestamps

def download_video(youtube_url, video_id, quality):
    ydl_opts = {"outtmpl": f"{video_id}.%(ext)s", "format": quality, "verbose": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            info = ydl.extract_info(youtube_url, download=False)
            
            # Check if the file exists and is not empty
            output_file = f"{video_id}.mp4"
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                raise Exception("Downloaded file is missing or empty")
            
            return info.get("duration", 0) * 1000
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        raise

def create_proxy_video(input_video, proxy_scale):
    proxy_video = input_video.replace(".mp4", "_proxy.mp4")
    cmd = [
        "ffmpeg",
        "-hwaccel", "auto",
        "-i", input_video,
        "-vf", f"scale=iw*{proxy_scale}:ih*{proxy_scale}",
        "-c:v", "h264_nvenc",
        "-preset", "fast",
        "-crf", "23",
        "-y", proxy_video
    ]
    try:
        run_subprocess_with_timeout(cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed: {e}")
        logger.error(f"FFmpeg output: {e.output}")
        raise
    return proxy_video
def auto_reframe_video(input_video, output_video, target_ratio=9/16, batch_size=7, proxy_scale=0.4, skip_frames=5):
    if not os.path.exists(input_video):
        logger.error(f"Input video not found: {input_video}")
        return

    proxy_video = create_proxy_video(input_video, proxy_scale)

    cap = cv2.VideoCapture(proxy_video)
    proxy_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    proxy_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize YOLO and DeepSort
    yolo = YOLO("yolov8m.pt")
    tracker = DeepSort(max_age=7, n_init=3, nn_budget=100, embedder="mobilenet")

    # Calculate new dimensions for the final output
    original_cap = cv2.VideoCapture(input_video)
    width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_cap.release()

    zoom_factor = 1.1
    if width / height < target_ratio:
        new_height = int(width / target_ratio)
        crop_height = int(new_height / zoom_factor)
        crop_width = int(width / zoom_factor)
    else:
        new_width = int(height * target_ratio)
        crop_width = int(new_width / zoom_factor)
        crop_height = int(height / zoom_factor)

    # Calculate proxy crop dimensions
    proxy_crop_width = int(crop_width * proxy_scale)
    proxy_crop_height = int(crop_height * proxy_scale)

    crop_positions = []
    last_crop_position = (0, 0)
    last_subject_position = (proxy_width // 2, proxy_height // 2)
    lost_subject_for = 0

    frames = []
    frame_count = 0

    def process_frame(frame):
        detections = yolo(frame)[0]
        people = []
        for data in detections.boxes.data.tolist():
            if data[5] == 0:  # class 0 is person
                xmin, ymin, xmax, ymax = map(int, data[:4])
                people.append([[xmin, ymin, xmax - xmin, ymax - ymin], 1.0, 0])

        tracks = tracker.update_tracks(people, frame=frame)
        return tracks

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (skip_frames + 1) == 0:
            frames.append(frame)

        frame_count += 1

        if len(frames) == batch_size or not ret:
            tracks_list = [process_frame(f) for f in frames]

            for frame, tracks in zip(frames, tracks_list):
                main_subject = None
                max_area = 0
                for track in tracks:
                    if track.is_confirmed():
                        bbox = track.to_tlbr()
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > max_area:
                            max_area = area
                            main_subject = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

                if main_subject:
                    subject = main_subject
                    last_subject_position = subject
                    lost_subject_for = 0
                else:
                    subject = last_subject_position
                    lost_subject_for += 1

                if lost_subject_for > fps:  # 1 second
                    subject = (proxy_width // 2, proxy_height // 2)

                def ease_camera(current, target, factor=0.2):
                    return int(current + (target - current) * factor)

                crop_x = ease_camera(last_crop_position[0], max(0, min(proxy_width - proxy_crop_width, subject[0] - proxy_crop_width // 2)))
                crop_y = ease_camera(last_crop_position[1], max(0, min(proxy_height - proxy_crop_height, subject[1] - proxy_crop_height // 2)))

                # Store the crop position (scaled back to original video size)
                original_crop_x = int(crop_x / proxy_scale)
                original_crop_y = int(crop_y / proxy_scale)
                crop_positions.append((original_crop_x, original_crop_y))

                last_crop_position = (crop_x, crop_y)

            frames = []

        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.2f}%)")

    cap.release()

    # Apply the crop positions to the original high-quality video
    cap = cv2.VideoCapture(input_video)
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_width, crop_height))

    frame_count = 0
    for crop_x, crop_y in crop_positions:
        for _ in range(skip_frames + 1):
            ret, frame = cap.read()
            if not ret:
                break

            cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
            out.write(cropped_frame)

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Exported {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.2f}%)")

    cap.release()
    out.release()

    # Clean up proxy video
    retry_file_operation(lambda: os.remove(proxy_video))

    logger.info(f"Auto-reframed video saved: {output_video}")

import subprocess

def run_subprocess_with_timeout(cmd, timeout=600, retries=3):  # 10 minutes timeout, 3 retries
    for attempt in range(retries):
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds (attempt {attempt + 1}/{retries}): {' '.join(cmd)}")
            if attempt == retries - 1:
                raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed (attempt {attempt + 1}/{retries}): {' '.join(cmd)}")
            logger.error(f"Error output: {e.stderr}")
            if attempt == retries - 1:
                raise
        time.sleep(10)  # Wait 10 seconds before retrying

def generate_short_video(video_id, timestamp, video_duration, output_dir, proxy_scale, skip_frames):
    start_time = max(0, timestamp - 15000)  # Start 15 seconds before the timestamp
    duration = 55000  # 45 seconds (30-60 second clip) - adjust as needed for your shorts
    video_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_dir, exist_ok=True)

    temp_output_video = os.path.join(video_dir, f"{video_id}_short_temp.mp4")
    final_output_video = os.path.join(video_dir, f"{video_id}_short_{timestamp}.mp4")

    source_video = f"{video_id}.mp4"
    if not os.path.exists(source_video):
        logger.error(f"Source video file not found: {source_video}")
        return

    try:
        # Extract the clip using GPU acceleration
        run_subprocess_with_timeout([
            "ffmpeg",
            "-hwaccel", "auto",
            "-i", source_video,
            "-ss", str(start_time / 1000),
            "-t", str(duration / 1000),
            "-c:v", "h264_nvenc",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            temp_output_video
        ], timeout=900)  # 15 minutes timeout for this step

        # Add a small delay before accessing the file
        time.sleep(1)

        # Auto-reframe the video
        auto_reframe_video(temp_output_video, final_output_video, proxy_scale=proxy_scale, skip_frames=skip_frames)

        # Add audio to the reframed video - no need to re-encode audio
        temp_audio = os.path.join(video_dir, f"{video_id}_short_temp_audio.m4a")
        run_subprocess_with_timeout([
            "ffmpeg",
            "-hwaccel", "auto",
            "-i", temp_output_video,
            "-vn",
            "-c:a", "copy",
            temp_audio
        ], timeout=300)  # 5 minutes timeout for audio extraction

        final_output_with_audio = os.path.join(video_dir, f"{video_id}_short_{timestamp}_with_audio.mp4")
        run_subprocess_with_timeout([
            "ffmpeg",
            "-hwaccel", "auto",
            "-i", final_output_video,
            "-i", temp_audio,
            "-c:v", "h264_nvenc",
            "-preset", "fast",
            "-c:a", "copy",
            "-shortest",
            final_output_with_audio
        ], timeout=600)  # 10 minutes timeout for final video creation

        # Clean up temporary files
        retry_file_operation(lambda: os.remove(temp_output_video))
        retry_file_operation(lambda: os.remove(temp_audio))
        retry_file_operation(lambda: os.remove(final_output_video))
        retry_file_operation(lambda: os.rename(final_output_with_audio, final_output_video))

        logger.info(f"Short video generated and auto-reframed: {final_output_video}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in generate_short_video: {e}")
        logger.error(f"FFmpeg output: {e.stdout}")
        logger.error(f"FFmpeg error: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error in generate_short_video: {e}")
def generate_short_videos(video_id, timestamps, video_duration, output_dir, num_workers, proxy_scale, skip_frames):
    start_time = 60000  # 1 minute
    end_time = video_duration - 120000  # 2 minutes
    filtered_timestamps = [ts for ts in timestamps if start_time <= ts <= end_time]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                generate_short_video,
                video_id,
                timestamp,
                video_duration,
                output_dir,
                proxy_scale,
                skip_frames,
            )
            for timestamp in filtered_timestamps
        ]

        for future in tqdm(futures, total=len(filtered_timestamps), desc="Generating short videos"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in generate_short_video: {e}")

def run_subprocess_with_timeout(cmd, timeout=300):  # 5 minutes timeout
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds: {' '.join(cmd)}")
        raise

def retry_file_operation(func, max_retries=5, delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise

def cleanup_temp_files(video_id):
    temp_files = [f"{video_id}.mp4", f"{video_id}_proxy.mp4"]
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file}: {e}")

def get_processed_timestamps(output_dir, video_id):
    video_dir = os.path.join(output_dir, video_id)
    if not os.path.exists(video_dir):
        return set()
    
    processed = set()
    for file in os.listdir(video_dir):
        if file.endswith('.mp4'):
            timestamp = int(file.split('_')[-1].split('.')[0])
            processed.add(timestamp)
    return processed

def main():
    try:
        args = parse_args()

        video_id = extract_video_id(args.youtube_url)

        video_duration = download_video(args.youtube_url, video_id, args.video_quality)

        most_replayed_data = get_most_replayed_parts(video_id)
        timestamps = find_top_replayed_timestamps(most_replayed_data, args.peak_prominence)

        processed_timestamps = get_processed_timestamps(args.output_dir, video_id)
        timestamps_to_process = [ts for ts in timestamps if ts not in processed_timestamps]

        if timestamps_to_process:
            generate_short_videos(
                video_id,
                timestamps_to_process,
                video_duration,
                args.output_dir,
                args.num_workers,
                args.proxy_scale,
                args.skip_frames,
            )
        else:
            logger.info("All timestamps have already been processed.")

        cleanup_temp_files(video_id)

    except Exception as e:
        logger.exception("An unexpected error occurred:")

if __name__ == "__main__":
    main()