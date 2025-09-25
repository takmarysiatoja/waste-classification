# Resources (based on):
# https://github.com/KomaliValluru/waste-classification/blob/main/waste-classification-vit.ipynb
# https://huggingface.co/google/vit-base-patch16-224-in21k


pip install --upgrade gradio
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state


labels = ['battery',
 'biological',
 'brown-glass',
 'cardboard',
 'clothes',
 'green-glass',
 'metal',
 'paper',
 'plastic',
 'shoes',
 'trash',
 'white-glass']



# Not needed
label2id = {}
id2label = {}
for i, class_name in enumerate(labels):
    label2id[class_name] = str(i)
    id2label[str(i)] = class_name


from transformers import ViTFeatureExtractor, ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    'watersplash/waste-classification',
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)



import gradio as gr
import torch


bin_colors = {
    "battery": "🟥🗑️",
    "biological": "🟫🗑️",
    "brown-glass": "🟩🗑️",
    "cardboard": "🟦🗑️",
    "clothes": "♻️🗑️",
    "green-glass": "🟩🗑️",
    "metal": "🟨🗑️",
    "paper": "🟦🗑️",
    "plastic": "🟨🗑️",
    "shoes": "♻️🗑️",
    "trash": "⬛🗑️",
    "white-glass": "🟩🗑️"
}
def predict(input):
    inputs = processor(images=input, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prob, predicted_class = torch.max(probs, 1)
    label = labels[predicted_class.item()]
    bin_color = bin_colors.get(label, "⬛🗑️")
    return f"{bin_color} {label} — {((prob.item()) * 100):.2f}%"


custom_css = """
body {
    background-color: #d0f0c0;
    font-family: 'Poppins', sans-serif;
    color: #2e7d32;
}

.gradio-container {
    background-color: #d0f0c0;
    border-radius: 20px;
    padding: 30px;
}

h1, h2, h3 {
    text-align: center;
    color: #1b5e20;
}

footer {
    display: none;
}

.gr-button {
    background-color: #81c784;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    font-size: 16px;
}

.gr-button:hover {
    background-color: #66bb6a;
}

textarea {
    font-size: 18px;
    padding: 10px;
    border-radius: 10px;
    border: 2px solid #a5d6a7;
    background-color: #f1f8e9;
}
"""


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload your trash photo"),
    outputs=gr.Textbox(label="Prediction", lines=2),
    title="♻️ Waste Classification AI",
    description="**Upload a photo** of any trash and get the category prediction!",
    article="Based on: [GitHub Repo](https://github.com/KomaliValluru/waste-classification-model-ViT)",
    theme="default",
    css=custom_css
)

demo.launch(share=True)
