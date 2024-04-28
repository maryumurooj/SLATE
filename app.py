import os
import subprocess
import re
from manim import *
import google.generativeai as genai
import sys
import webbrowser
from prompt_generator import generate_universal_prompt
from manim_examples import examples
# Set the path for FFmpeg directly in the Manim configuration
config.ffmpeg_executable = r"C:\ffmpeg-2024-04-18-git-35ae44c615-essentials_build\bin\ffmpeg.exe"

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

def fetch_and_display_content(user_prompt):
    """Process user prompt to fetch and display both textual explanation and related Manim code."""
    # First interaction with Gemini API for a textual explanation
    response_text = model.generate_content(user_prompt)
    print("\nTextual Explanation from Gemini:")
    print(response_text.text)  # Display the textual response in the terminal

    # Extract keyword from the prompt, assume it's the first word for simplicity
    keyword = user_prompt.split()[0].lower()
    manim_script = examples.get(keyword, None)
    if not manim_script:
        print(f"No Manim script found for the keyword: {keyword}")
        return

    # Second interaction with Gemini API using the found Manim script
    combined_prompt = f"referring this{manim_script}\n\n change for {user_prompt}"
    response_manim = model.generate_content(combined_prompt)
    print("\nManim Code Response from Gemini:")
    print(response_manim.text)  # Display the Manim response in the terminal

if __name__ == '__main__':
    user_prompt = input("Enter your prompt: ")
    fetch_and_display_content(user_prompt)

def fetch_and_display_content(user_prompt):
    """Fetches both textual explanation and Manim code, then renders and displays video."""
    full_prompt = generate_universal_prompt(user_prompt)
    response = model.generate_content(full_prompt)
    print("Raw API Response:", response.text)  # Debug: Print the raw API response

    textual_content = response.text.strip()  # Assuming the API returns text directly
    if textual_content:
        to_markdown(textual_content)
    else:
        print("No textual content was found.")

    # Fetch and tune the Manim script
   

def render_and_display_manim(manim_code, filename="ManimScene"):
    """Renders Manim code and displays the generated video."""
    base_dir = r"C:\Users\HUSSAIN'S\Desktop\SLATE"
    media_dir = os.path.join(base_dir, "media", "videos")
    script_filename = os.path.join(media_dir, f"{filename}.py")
    os.makedirs(media_dir, exist_ok=True)
    with open(script_filename, "w") as script_file:
        script_file.write(manim_code)
    try:
        subprocess.run(['manim', '-pm', '--media_dir', media_dir, '-o', filename, script_filename], check=True)
        video_filename = os.path.join(media_dir, f"{filename}.mp4")
        print(f"Rendered video path: {video_filename}")
        webbrowser.open(video_filename)
    except subprocess.CalledProcessError as e:
        print(f"Failed to render Manim video. Here's the error:\n{e}")

