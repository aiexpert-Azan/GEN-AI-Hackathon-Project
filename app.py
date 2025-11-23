import os
import base64
import html
import gradio as gr
import numpy as np
from PIL import Image
from groq import Groq
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# -------------------------
# Load environment key
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Add it in Hugging Face → Settings → Secrets.")

client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# Load models
# -------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip.eval()

if torch.cuda.is_available():
    model_blip.to("cuda")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Load data + FAISS index
# -------------------------
DATA_CSV = "artifacts/products.csv"
META_PKL = "artifacts/products_meta.pkl"
FAISS_INDEX = "artifacts/products.index"
EMB_PATH = "artifacts/product_embeddings.npy"

df = pd.read_pickle(META_PKL)
index = faiss.read_index(FAISS_INDEX)


# -------------------------
# Utility Functions
# -------------------------
def caption_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(model_blip.device)
    with torch.no_grad():
        out = model_blip.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)


def embed_text(text):
    return embedder.encode(text, normalize_embeddings=True).astype("float32")


def retrieve_by_text(query, top_k=5):
    query_vec = embed_text(query).reshape(1, -1)
    D, I = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        row = df.iloc[idx].to_dict()
        row["score"] = float(score)
        results.append(row)

    return results


def make_chat_messages(question, items):
    ctx = []
    for i, it in enumerate(items, 1):
        ctx.append(f"Item {i}: {it['title']} — {it['description']} — Price: ${it['price']}")

    context_block = "\n".join(ctx)

    system = {
        "role": "system",
        "content": "You are an AI shopping assistant. Use ONLY the given product context to answer clearly and briefly."
    }
    user = {
        "role": "user",
        "content": f"Context:\n{context_block}\n\nQuestion: {question}"
    }
    return [system, user]


def call_groq(messages):
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"[Groq API error] {e}"


def format_product_cards(items):
    cards = []
    for it in items:
        path = it.get("filepath", "")
        img_tag = ""

        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img_tag = f"""
            <img src="data:image/jpeg;base64,{b64}"
            style="width:180px;height:140px;object-fit:cover;border-radius:8px;" />
            """

        title = html.escape(str(it["title"]))
        price = html.escape(str(it["price"]))
        score = f"{it['score']:.3f}"

        card = f"""
        <div style="border:1px solid #ddd;padding:10px;border-radius:12px;
        width:200px;margin:6px;background:white;
        box-shadow:0 2px 6px rgba(0,0,0,0.1);">

            {img_tag}
            <div style="font-weight:600;margin-top:6px;">{title}</div>
            <div style="color:#005bbb;font-weight:700;margin-top:4px;">${price}</div>
            <div style="font-size:12px;color:#666;margin-top:4px;">score: {score}</div>
        </div>
        """
        cards.append(card)

    return "<div style='display:flex;flex-wrap:wrap;'>" + "".join(cards) + "</div>"


# -------------------------
# Main Pipeline
# -------------------------
def pipeline(image, question):
    try:
        if image is None:
            return "❌ Please upload an image.", "", ""

        caption = caption_image(image)
        retrieved = retrieve_by_text(caption, top_k=5)
        cards = format_product_cards(retrieved)
        messages = make_chat_messages(question, retrieved)
        answer = call_groq(messages)

        return caption, cards, answer

    except Exception as e:
        return f"[Error] {e}", "", ""


# -------------------------
# Gradio UI
# -------------------------
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #f0f4ff, #e8faff);
}
.box {
    padding: 18px;
    border-radius: 16px;
    background: white;
    border: 1px solid #d9e2ec;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
        <h1 style="text-align:center;">
            🛍️ Multimodal AI Shopping Assistant
        </h1>
        <p style="text-align:center;">Upload an image → AI captions → retrieves similar products → answers your question.</p>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="box"):
            img = gr.Image(type="pil", label="Upload product image")
            txt = gr.Textbox(value="Show me similar products under $100")
            btn = gr.Button("🔍 Search")

        with gr.Column(scale=1, elem_classes="box"):
            caption_out = gr.Textbox(label="Caption", interactive=False)
            products_out = gr.HTML()
            answer_out = gr.Textbox(label="AI Response", interactive=False)

    btn.click(pipeline, inputs=[img, txt], outputs=[caption_out, products_out, answer_out])

demo.launch()
