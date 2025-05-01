import os
from flask import Flask, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

model = None
processor = None

def load_model_and_processor():
    global model, processor
    print("Loading model and processor...")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Model and processor loaded.")

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "BLIP Flask API is live!"})

@app.route("/generate", methods=["POST"])
def generate():
    if model is None or processor is None:
        return jsonify({"error": "Model or processor not loaded."}), 500

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image file provided."}), 400

    image_path = "temp_image.jpg"
    file.save(image_path)

    caption = generate_caption(image_path)
    os.remove(image_path)

    return jsonify({"caption": caption})

if __name__ == "__main__":
    load_model_and_processor()
    app.run(host="0.0.0.0", port=10000)
