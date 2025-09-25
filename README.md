# ♻️ Waste Classification AI

A simple web app that classifies waste images into categories such as **plastic, glass, paper, metal, clothes, shoes, etc.**  
The app also suggests the correct bin color for disposal.  
Built with [Gradio](https://gradio.app/) and [HuggingFace Transformers](https://huggingface.co/).


## 📂 Project Structure
waste-classification/
│
├── app/
│ └── app.py # Main Gradio app
│
├── notebooks/
│ └── exploration.ipynb # (optional) Jupyter notebook with tests
│
├── requirements.txt # Project dependencies
├── README.md # Documentation
├── .gitignore # Files to ignore in Git
├── LICENSE # License (MIT)
└── screenshot.png # Example screenshot


## 🚀 Demo
Upload a photo of trash and get the predicted category along with a confidence score.



## 🛠️ Installation
Clone the repository and install the required packages:

```bash
git clone https://github.com/YOUR_USERNAME/waste-classification.git
cd waste-classification
pip install -r requirements.txt
```


## ▶️ Run the App
Run the Gradio interface:

```bash
cd app
python app.py
```


## 📊 Model
This project uses a Vision Transformer (ViT) model fine-tuned for waste classification.
It predicts categories such as:
battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass.

The model assigns each input image to one of the categories and maps it to the correct bin.



## 📚 Resources
- [HuggingFace Model – watersplash/waste-classification](https://huggingface.co/watersplash/waste-classification)
- [Vision Transformer Base – google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- [GitHub Reference Project](https://github.com/KomaliValluru/waste-classification)


