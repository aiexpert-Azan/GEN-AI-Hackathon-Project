**🛍️ Multimodal RAG for ecommerce product Assistance**

Image → Caption → Vector Search → AI Answering

This Space lets you upload any product image, automatically generate a caption, retrieve visually similar products, and then ask a natural-language question that the AI answers using the retrieved product context.

Built using:

BLIP for image captioning

Sentence Transformers for embeddings

FAISS for fast similarity search

Groq LLM API for chat responses

Gradio for the UI

**🚀 How it Works**

Upload an image
The BLIP model generates a high-quality caption from the image.

Semantic Search
The caption is converted to an embedding (all-MiniLM-L6-v2).
A FAISS index matches it with the top similar products.

Product Display
Retrieved items appear as product cards with:

image

title

price

similarity score

Ask a Question
A question like:
“Show me budget options”
or
“What’s similar but cheaper?”

→ The Groq LLM answers using ONLY the retrieved products.

🧩 Project Structure
/
├── app.py                         # Main Gradio app
├── requirements.txt               # Dependencies (CPU-friendly)
├── README.md                      # Project documentation
├── artifacts/                     # All search-related data
│   ├── products.csv
│   ├── products_meta.pkl
│   ├── products.index            
│   ├── product_embeddings.npy
│   └── (any other metadata files)
├── pics_products/            
│   ├── img_001.jpg
│   ├── img_002.jpg
│   ├── img_003.jpg
│   └── ... (all product images here)


**🛠️ Dependencies**

torch (CPU build)

sentence-transformers

transformers

faiss-cpu

gradio 4.x

groq

Your requirements.txt is configured to match HuggingFace space limits.

**🖼️ Models Used
Image Captioning**

Salesforce/blip-image-captioning-base

Text Embedding

sentence-transformers/all-MiniLM-L6-v2

Vector Search

FAISS Index built from product_embeddings.npy

Chat Model

llama-3.1-8b-instant via Groq API

**▶️ Running Locally**

Install dependencies:

pip install -r requirements.txt


Set your key:

export GROQ_API_KEY="your-key"


Run:

python app.py

**🙌 Credits**

Salesforce Research – BLIP

HuggingFace Transformers & Sentence Transformers

Facebook AI Research – FAISS

Groq – LLM API

Gradio – UI framework
