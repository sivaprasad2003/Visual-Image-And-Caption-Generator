# AI Image Processing App

## 📌 Overview
This AI Image Processing App provides functionalities for:
- **Image Captioning**: Automatically generating captions for uploaded images.
- **Text-to-Image Generation**: Creating images from text prompts using Stable Diffusion and DALL·E 3.

## 🚀 Features
- **BLIP Image Captioning** for high-quality image descriptions.
- **Stable Diffusion & DALL·E 3 Integration** for AI-generated images.
- **Streamlit UI** for an interactive user experience.

## 📂 Project Structure
```
📁 Visual-Image-And-Caption-Generator/
├── app.py                     # Main application script
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── blip-image-captioning-base/                   # BLIP model directory (download manually)
```

## 🔧 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sivaprasad2003/Visual-Image-And-Caption-Generator.git
cd AI-Image-Processing-App
```

### 2️⃣ Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Download BLIP Image Captioning Model
You need to manually download the **BLIP Image Captioning Base** model from Hugging Face:
```bash
pip install huggingface_hub
huggingface-cli download Salesforce/blip-image-captioning-base --local-dir models/blip-image-captioning-base
```

Alternatively, you can download it manually from:
[https://huggingface.co/Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
and place it inside the `models/` directory.

## ▶️ Run the Application from terminal

Open terminal from the existing folder

streamlit run app.py


## 📜 Requirements
Download required PIP from Library mentioned in a `requirements.txt`

## 🤝 Contributing
Feel free to fork this project, submit pull requests, or report issues!

## 📜 License
MIT License

