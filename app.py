import os
import webbrowser
import re
from manim import *
import google.generativeai as genai

# Set the path for FFmpeg directly in the Manim configuration
config.ffmpeg_executable = r"C:\ffmpeg-2024-04-18-git-35ae44c615-essentials_build\bin\ffmpeg.exe"

# Function to format text as Markdown (for display in console or Markdown-supported environments)
def to_markdown(text):
    """Imitate Markdown formatting in console output"""
    print("\nFormatted as Markdown:\n")
    print(text)

# Configure API key and model
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def adjust_manim_code(code):
    """Adjusts generated Manim code to ensure it is syntactically correct."""
    if "from manim import *" not in code:
        code = "from manim import *\n" + code
    code = re.sub(r"Circle\((.*?)\)", r"Circle(radius=\1)", code)
    return code

def parse_response(response):
    """Parses the response to extract the Manim code section."""
    parts = response.split('```')
    return parts[1].strip() if len(parts) > 1 else ""

def generate_and_display_response(user_prompt):
    """Generates Manim code based on user prompt using the Gemini API"""
    full_prompt = f"Generate a detailed Manim script for: {user_prompt}"
    response = model.generate_content(full_prompt)
    text_response = response.text
    to_markdown(text_response)  # Display textual explanation
    manim_code = parse_response(text_response)
    return manim_code

def render_and_display_manim(manim_code, filename="ManimScene"):
    """Renders Manim code and displays the generated video"""
    root_dir = "C:\\Users\\HUSSAIN'S\\Desktop\\SLATE"
    script_filename = os.path.join(root_dir, f"{filename}.py")
    media_dir = os.path.join(root_dir, "media", "videos")

    manim_code = adjust_manim_code(manim_code)
    with open(script_filename, "w") as script_file:
        script_file.write(manim_code)

    os.makedirs(media_dir, exist_ok=True)
    os.system(f"manim -pql --media_dir {media_dir} -o {filename} {script_filename}")

    video_filename = os.path.join(media_dir, f"{filename}.mp4")
    print(f"Rendered video path: {video_filename}")
    webbrowser.open(video_filename)

if __name__ == '__main__':
    user_prompt = input("Enter your prompt: ")
    manim_code = generate_and_display_response(user_prompt)
    if manim_code:
        render_and_display_manim(manim_code)
    else:
        print("No Manim code was provided or generated correctly.")
