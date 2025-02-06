import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


model_name = "dima806/deepfake_vs_real_image_detection"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE)

def predict(input_image: Image.Image):
   
    inputs = processor(images=input_image, return_tensors="pt").to(DEVICE)


    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    
    predicted_class = logits.argmax(-1).item()
    confidence = probabilities[0][predicted_class].item()

   
    class_names = ["real", "fake"]
    predicted_label = class_names[predicted_class]

 
    confidences = {
        class_names[i]: float(prob) for i, prob in enumerate(probabilities[0])
    }

    return confidences, input_image


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=[
        gr.Label(label="Classification"),
        gr.Image(type="pil", label="Input Image")
    ],
    title="Deepfake Detection",
    description="Upload an image to classify it as real or fake using the dima806/deepfake_vs_real_image_detection model."
) 

if __name__ == "__main__":
    interface.launch()