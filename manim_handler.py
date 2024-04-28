import os
import subprocess
import webbrowser
from manim import config

def extract_manim_code(api_response):
    """Extracts Manim code from API response, assuming code is enclosed in '```python' and '```'."""
    start_marker = '```manim'
    end_marker = '```'
    start_idx = api_response.find(start_marker) + len(start_marker)
    end_idx = api_response.find(end_marker, start_idx)
    if start_idx != -1 and end_idx != -1:
        return api_response[start_idx:end_idx].strip()
    return None

def save_and_render_manim_code(manim_code, filename="ManimScene"):
    """Save the Manim code to a file and render it."""
    script_path = os.path.join(os.getcwd(), f"{filename}.py")
    with open(script_path, 'w') as file:
        file.write(manim_code)
    print(f"Manim script saved at: {script_path}")

    # Render and display the video
    media_dir = os.path.join(os.getcwd(), "media")
    subprocess.run(['manim', '-pql', '--media_dir', media_dir, '-o', filename, script_path], check=True)
    video_path = os.path.join(media_dir, "videos", filename + ".mp4")
    print(f"Video rendered at: {video_path}")
    webbrowser.open(video_path)
