import os
import subprocess
import re
from manim import *
import google.generativeai as genai
import sys
import webbrowser

# Configuration for FFmpeg, essential for video rendering in Manim
config.ffmpeg_executable = r"C:\ffmpeg-2024-04-18-git-35ae44c615-essentials_build\bin\ffmpeg.exe"

# Load the Google API key from environment variables and configure the APIs
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
text_model = genai.GenerativeModel('gemini-pro')
visual_model = genai.GenerativeModel('vertex-custom-manim')  # Hypothetical model for Manim scripts

def to_markdown(text):
    """Formats and displays text in a console as Markdown-styled output."""
    print("\n**Formatted as Markdown:**\n")
    print(text)

def adjust_manim_code(code):
    """Ensures that the generated Manim code is ready for execution."""
    if "from manim import *" not in code:
        code = "from manim import *\n" + code
    code = re.sub(r"Circle\((.*?)\)", r"Circle(radius=\1)", code)
    return code

def fetch_and_interact(user_prompt):
    """Fetches textual explanation and optionally generates and displays Manim video based on user interaction."""
    try:
        # Fetching textual explanation
        text_response = text_model.generate_content(user_prompt)
        to_markdown(text_response.text)

        # Asking user for visual representation
        user_choice = input("Would you like a visual representation of this? (Yes/No): ").strip().lower()
        if user_choice == 'yes':
            visual_prompt = f"Explain and show an animation of the {user_prompt}"
            manim_response = visual_model.generate_content(visual_prompt)
            manim_code = adjust_manim_code(manim_response.text)
            if manim_code:
                render_and_display_manim(manim_code)
            else:
                print("No valid Manim code was generated.")
        else:
            print("No visual representation requested.")
    except Exception as e:
        print(f"An error occurred: {e}")

def render_and_display_manim(manim_code, filename="ManimScene"):
    """Renders Manim code into a video and opens it for viewing."""
    root_dir = "C:\\Users\\HUSSAIN'S\\Desktop\\SLATE"
    script_filename = os.path.join(root_dir, f"{filename}.py")
    media_dir = os.path.join(root_dir, "media", "videos")

    with open(script_filename, "w") as script_file:
        script_file.write(manim_code)

    os.makedirs(media_dir, exist_ok=True)
    subprocess.run(['manim', '-pql', '--media_dir', media_dir, '-o', filename, script_filename], check=True)
    video_filename = os.path.join(media_dir, f"{filename}.mp4")
    print(f"Rendered video path: {video_filename}")
    webbrowser.open(video_filename)

if __name__ == '__main__':
    user_prompt = input("Enter your prompt: ")
    fetch_and_interact(user_prompt)
