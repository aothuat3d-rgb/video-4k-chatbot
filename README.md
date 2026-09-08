# Video 4K + Watermark Removal Chatbot

Local web app for uploading a video, removing a fixed-position watermark, and upscaling the result to 4K.

## Requirements
- Python 3.10+
- FFmpeg available as `ffmpeg` in PATH

## Run
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000

The app uses OpenCV inpainting for the watermark and FFmpeg Lanczos for 4K upscaling.
For best results, upload a small watermark crop and use Auto detect. Manual coordinates are also available.
Only use it for videos you own or have permission to edit.
