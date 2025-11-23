import os
import base64
import html
import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import torch
import faiss
from sentence_transformers import SentenceTransformer, util
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq

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


df = pd.read_csv("artifacts/products.csv")

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings for product titles
embeddings = embedder.encode(df['title'].tolist(), normalize_embeddings=True).astype("float32")

# Save metadata
meta = {
    "product_ids": df['id'].tolist(),
    "embeddings": embeddings
}

with open("artifacts/products_meta.pkl", "wb") as f:
    pickle.dump(meta, f, protocol=4)

index = faiss.read_index("artifacts/products.index")

# -------------------------
# Utility Functions
# -------------------------
def caption_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    device = model_blip.device
    inputs = {k: v.to(device) for k, v in inputs.items()}  # move all tensors to device
    with torch.no_grad():
        out = model_blip.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)

def embed_text(text):
    return embedder.encode(text, normalize_embeddings=True).astype("float32")

def retrieve_by_text(query, top_k=3):
    query_vec = embed_text(query).reshape(1, -1)
    D, I = index.search(query_vec, top_k)
    items = []
    for score, idx in zip(D[0], I[0]):
        row = df.iloc[idx].to_dict()  # use df, not df_products
        row["score"] = float(score)
        items.append(row)
    return items

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

# -------------------------
# Format product cards without BASE_DIR
# -------------------------

def format_product_cards(items):
    cards = []
    for it in items:
        relative_path = it.get("filepath", "")  # Already points to pics_products/filename
        img_tag = ""
        if os.path.exists(relative_path):
            try:
                with open(relative_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                img_tag = f"""
                <img src="data:image/jpeg;base64,{b64}"
                style="width:180px;height:140px;object-fit:cover;border-radius:8px;" />
                """
            except:
                pass
        title = html.escape(str(it.get("title", "")))
        price = html.escape(str(it.get("price", "")))
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
# Main Pipeline without BASE_DIR
# -------------------------
def pipeline(img: Image.Image, query: str):
    try:
        # 1️⃣ Caption
        caption = caption_image(img)

        # 2️⃣ Retrieve similar products
        items = retrieve_by_text(caption, top_k=3)

        # 3️⃣ AI Response
        messages = make_chat_messages(query, items)
        answer = call_groq(messages)

        # 4️⃣ Products HTML
        products_html = format_product_cards(items)

        return caption, products_html, answer

    except Exception as e:
        print("🚨 Pipeline error:", e)
        return f"ERROR: {e}", "<p style='color:red;'>Failed to retrieve products</p>", f"ERROR: {e}"

# -------------------------
# Custom CSS
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
/* Make AI Response box bigger */
.response-box .gr-textbox {
    min-height: 200px;  /* increase height */
    font-size: 16px;    /* larger text */
    padding: 12px;
}
"""
# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:
    gr.HTML(f"<style>{custom_css}</style>")

    gr.HTML("""
    <h1 style="text-align:center;">🛍️ Multimodal AI Shopping Assistant</h1>
    <p style="text-align:center;">Upload an image → AI captions → retrieves similar products → answers your question.</p>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="box"):
            img_input = gr.Image(type="pil", label="Upload product image")
            txt_input = gr.Textbox(value="Show me similar products under $100", label="Your query")
            btn = gr.Button("🔍 Search")

        with gr.Column(scale=1, elem_classes="box"):
            caption_out = gr.Textbox(label="Caption", interactive=False)
            products_out = gr.HTML(label="Products")
            answer_out = gr.Textbox(label="AI Response", interactive=False, elem_classes="response-box")

    btn.click(pipeline, inputs=[img_input, txt_input], outputs=[caption_out, products_out, answer_out])

# -------------------------
# Launch app
# -------------------------
demo.launch()
