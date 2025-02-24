import streamlit as st
import torch
import os
import io
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
import torch.nn as nn
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
from openai import OpenAI

# ---------------------- Configuration ---------------------- #
st.set_page_config(page_title="Visual Image And Caption Generator", layout="wide")

MODEL_DIRECTORY = "Your Model Directory"  # Replace with your model directory
OPENAI_API_KEY = "Your OpenAI API key"  # Replace with your OpenAI API key
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------- Model Loading ---------------------- #

if not os.path.exists(MODEL_DIRECTORY):
    st.error("Error: Model directory not found. Please download the model first.")
else:
    processor = BlipProcessor.from_pretrained(MODEL_DIRECTORY)
    config = BlipConfig.from_pretrained(MODEL_DIRECTORY)

    class CustomBlipForConditionalGeneration(BlipForConditionalGeneration):
        def __init__(self, config):
            super().__init__(config)
            self.additional_layer1 = nn.Linear(768, 768)
            self.additional_layer2 = nn.Linear(768, 768)
            self.init_weights()

        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            sequence_output = outputs[0]
            sequence_output = self.additional_layer1(sequence_output)
            sequence_output = nn.functional.relu(sequence_output)
            sequence_output = self.additional_layer2(sequence_output)
            return sequence_output, outputs[1:]

    model = CustomBlipForConditionalGeneration.from_pretrained(MODEL_DIRECTORY)

@st.cache_resource
def load_text_to_image_model():
    return DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda" if torch.cuda.is_available() else "cpu")

sd_pipeline = load_text_to_image_model()

# ---------------------- DALL·E 3 Image Generation ---------------------- #
def generate_image_with_dalle(prompt):
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        return Image.open(io.BytesIO(image_response.content))
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# ---------------------- Streamlit UI ---------------------- #
st.title("🖼️ AI Image Processing App")
st.sidebar.header("Choose a Function:")
option = st.sidebar.radio("", ["Image Captioning", "Text-to-Image Generation"])

# ---------------------- Image Captioning ---------------------- #
if option == "Image Captioning":
    st.header("📸 Image Captioning")
    uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        text_input = st.text_input("Enter text prefix for captioning (optional):", "")
        caption_count = st.number_input("Number of captions to generate:", min_value=1, max_value=10, value=3)

        if st.button("Generate Captions"):
            inputs = processor(image, text_input, return_tensors="pt")
            captions = []
            for _ in range(caption_count):
                out = model.generate(
                    **inputs,
                    max_length=50,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.8
                )
                captions.append(processor.decode(out[0], skip_special_tokens=True))

            st.success("Generated Captions:")
            for idx, cap in enumerate(captions, 1):
                st.write(f"{idx}. {cap}")

# ---------------------- Text-to-Image Generation ---------------------- #
elif option == "Text-to-Image Generation":
    st.header("🖍️ Text-to-Image Generation")
    model_choice = st.radio("Choose Image Generation Model:", ["Stable Diffusion", "DALL·E 3"])
    prompt = st.text_area("Enter a description for the image you want to generate:")
    
    if st.button("Generate Image"):
        if not prompt:
            st.error("Please enter a description.")
        else:
            with st.spinner("Generating Image... This may take a few seconds."):
                if model_choice == "Stable Diffusion":
                    image = sd_pipeline(prompt).images[0]
                elif model_choice == "DALL·E 3":
                    image = generate_image_with_dalle(prompt)
                
                if image:
                    st.image(image, caption="Generated Image", use_column_width=True)
                    image.save("generated_image.png")
                    st.download_button("Download Image", data=open("generated_image.png", "rb"), file_name="generated_image.png", mime="image/png")

st.sidebar.markdown("---")
st.sidebar.write("Developed by **SP** 🚀")
