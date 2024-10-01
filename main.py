import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

system_prompt = """You are a helpful assistant to help me generate stable diffusion image generation prompts.
Your task is make stable diffusion image generation prompt for generate exactly same image as the input.
While generate image generation prompt, please make sure you generate highly realistic image generation prompts for real estate images.
The assets of the generate image should be same (i.e. general amenities).
Do not include anything in the response except image generation prompt.
No system message, no guidance, no explanation etc.

This is common real estate image generation prompt example.
real estate photography style {prompt} . professional, inviting, well-lit, high-resolution, property-focused, commercial, highly detailed
"""

def generate_real_estate_prompt(image_path):
    """Generates a Stable Diffusion prompt for a real estate image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The generated Stable Diffusion prompt.
    """
    image_file = genai.upload_file(path=image_path, display_name="real estate image")
    response = model.generate_content([image_file, system_prompt])
    return response.text

# Example usage
image_path = "data/1.jpg"
prompt = generate_real_estate_prompt(image_path)
print(f"Generated Prompt: {prompt}")
