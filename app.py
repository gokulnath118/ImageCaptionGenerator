import os
import io
import base64
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configuration for generating captions
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to predict captions
def predict_step(images):
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# Helper function to convert PIL image to base64 string
def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return redirect(request.url)

        images = []
        image_strs = []
        for file in files:
            img = Image.open(file)
            if img.mode != "RGB":
                img = img.convert(mode="RGB")
            images.append(img)
            image_strs.append(pil_to_base64(img))

        captions = predict_step(images)
        results = zip(image_strs, captions)
        return render_template('index.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
