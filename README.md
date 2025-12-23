# Image Caption + VQA + Explanation (Streamlit)

This repository is a Streamlit version of a Colab/Gradio demo that:
1. Generates an **image caption** using **BLIP** (`Salesforce/blip-image-captioning-base`)
2. Optionally answers a question about the image (**VQA**) using **BLIP VQA** (`Salesforce/blip-vqa-base`)
3. Produces a short **3–5 sentence explanation** (optional)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (get a shareable link)
1. Push this repo to GitHub.
2. Streamlit Community Cloud → **New app**
3. Select your repo and set:
   - **Main file path:** `app.py`
4. Click **Deploy**.

Your link will look like:
`https://<your-app-name>.streamlit.app`

## Notes on resource limits
Streamlit Community Cloud has resource limits that can be hit by large models.
This Streamlit version uses **Flan-T5 Small** as an optional explainer because it is much lighter than a 1.1B+ LLM.
If you want to use a larger model (like TinyLlama), deploy on a GPU host instead.

## Files
- `app.py` — Streamlit UI + model loading + inference
- `original_gradio_colab.py` — original script (reference)
- `requirements.txt` — dependencies
