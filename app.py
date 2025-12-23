import numpy as np
import streamlit as st
from PIL import Image

import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(
    page_title="BLIP Caption + VQA Studio",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ BLIP Vision Explanation Studio (Streamlit)")
st.caption("Upload an image, optionally ask a question, and get a caption, a short answer, and an explanation.")

# ----------------------------
# Device
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Running on: **{DEVICE.upper()}**")

# ----------------------------
# Model IDs (Hugging Face)
# ----------------------------
CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
VQA_MODEL_ID = "Salesforce/blip-vqa-base"

# Small instruction model for explanation (more Cloud-friendly than a 1.1B+ LLM)
EXPLAINER_MODEL_ID = "google/flan-t5-small"


# ----------------------------
# Caching: load models once per server process
# ----------------------------
@st.cache_resource(show_spinner="Loading BLIP caption model...")
def load_caption():
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading BLIP VQA model...")
def load_vqa():
    processor = BlipProcessor.from_pretrained(VQA_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(VQA_MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading explainer model (Flan-T5 Small)...")
def load_explainer():
    tok = AutoTokenizer.from_pretrained(EXPLAINER_MODEL_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(EXPLAINER_MODEL_ID).to(DEVICE)
    mdl.eval()
    return tok, mdl


def preprocess_pil(image) -> Image.Image:
    """Ensure a reasonably sized PIL RGB image."""
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    max_side = 512
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        image = image.resize((int(w * scale), int(h * scale)))
    return image


def get_blip_caption(image: Image.Image, max_length=30, num_beams=5) -> str:
    processor, model = load_caption()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def get_blip_vqa_answer(image: Image.Image, question: str, max_length=30, num_beams=5) -> str:
    processor, model = load_vqa()
    prompt = f"Question: {question} Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    text = processor.decode(out[0], skip_special_tokens=True).strip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text


def explain_with_small_llm(caption: str, qa_answer: str | None, user_question: str) -> str:
    """Generate 3–5 simple sentences using a small instruction model."""
    tok, mdl = load_explainer()

    prompt = (
        "You are an expert visual tutor. Explain the image for a non-expert in 3 to 5 simple sentences.\n"
        f"Caption: {caption}\n"
    )
    if qa_answer:
        prompt += f"Short answer: {qa_answer}\n"
    if user_question and user_question.strip():
        prompt += f"User question: {user_question.strip()}\n"
    prompt += "Explanation:"

    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=140)
    text = tok.decode(out[0], skip_special_tokens=True).strip()
    if "Explanation:" in text:
        text = text.split("Explanation:")[-1].strip()
    return text


def explain_template(caption: str, qa_answer: str, user_question: str) -> str:
    """Fallback explanation (no extra model)."""
    if user_question and user_question.strip():
        return (
            f"This image can be summarized as: {caption}. "
            f"For your question ('{user_question.strip()}'), the best short answer is: {qa_answer}. "
            "Overall, the scene appears consistent with the caption and answer above."
        )
    return (
        f"This image can be summarized as: {caption}. "
        "It shows the main objects and context described in the caption."
    )


# ----------------------------
# UI
# ----------------------------
st.sidebar.header("Options")
use_vqa = st.sidebar.checkbox("Enable VQA (loads a second BLIP model)", value=True)
use_llm = st.sidebar.checkbox("Generate detailed explanation with a small LLM", value=False)

with st.sidebar.expander("Generation parameters", expanded=False):
    num_beams = st.slider("num_beams (quality vs speed)", 1, 10, 5, 1)
    max_len = st.slider("max_length (caption/answer)", 10, 60, 30, 5)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    question = st.text_input("Optional question (for VQA)", placeholder="e.g., What is the person doing? Indoors or outdoors?")
    analyze = st.button("Analyze", type="primary", use_container_width=True)

with col2:
    st.subheader("Output")
    caption_box = st.empty()
    answer_box = st.empty()
    explanation_box = st.empty()

if analyze:
    if uploaded is None:
        st.error("Please upload an image first.")
        st.stop()

    img = Image.open(uploaded)
    img = preprocess_pil(img)

    st.image(img, caption="Uploaded image (downscaled if needed)", use_container_width=True)

    with st.spinner("Generating caption..."):
        try:
            caption = get_blip_caption(img, max_length=max_len, num_beams=num_beams)
        except Exception as e:
            caption = f"Error generating caption: {e}"

    caption_box.text_area("BLIP Caption", caption, height=90)

    qa_answer = "VQA disabled."
    if use_vqa and question.strip():
        with st.spinner("Answering question (VQA)..."):
            try:
                qa_answer = get_blip_vqa_answer(img, question.strip(), max_length=max_len, num_beams=num_beams)
            except Exception as e:
                qa_answer = f"Error generating answer: {e}"
    elif use_vqa and not question.strip():
        qa_answer = "No specific question asked."
    answer_box.text_area("BLIP Short Answer", qa_answer, height=90)

    with st.spinner("Generating explanation..."):
        try:
            if use_llm:
                explanation = explain_with_small_llm(
                    caption=caption,
                    qa_answer=None if ("Error" in qa_answer or "disabled" in qa_answer.lower() or "No specific" in qa_answer) else qa_answer,
                    user_question=question,
                )
            else:
                explanation = explain_template(caption, qa_answer, question)
        except Exception as e:
            explanation = f"Error generating explanation: {e}"

    explanation_box.text_area("Explanation", explanation, height=220)

st.divider()
st.caption(
    "Note: On first run, the app downloads models from the Hugging Face Hub. This can take a while. "
    "Streamlit caching keeps models in memory across reruns for better performance."
)
