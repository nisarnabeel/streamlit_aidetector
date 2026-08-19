import numpy as np
import pandas as pd
import streamlit as st
from PIL import ExifTags, Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

st.write("""
# AI-Generated Image Detector

Upload one or more images and this app will estimate whether each is a real
photo or AI-generated (e.g. Midjourney / Stable Diffusion / DALL-E).
""")

st.caption(
    "⚠️ This is a rough, best-effort estimate from small open-source models, "
    "not a reliable verdict. It can be wrong, especially on newer generators, "
    "compressed/re-uploaded images, or heavily edited real photos."
)

MODEL_NAMES = ["Organika/sdxl-detector", "haywoodsloan/ai-image-detector-deploy"]
SALIENCY_MODEL_NAME = "haywoodsloan/ai-image-detector-deploy"


@st.cache_resource
def load_detectors():
    return [pipeline("image-classification", model=name) for name in MODEL_NAMES]


@st.cache_resource
def load_saliency_model():
    processor = AutoImageProcessor.from_pretrained(SALIENCY_MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(SALIENCY_MODEL_NAME)
    model.eval()
    return processor, model


def artificial_score(results):
    for r in results:
        if r["label"].lower() == "artificial":
            return r["score"]
    return None


def check_exif(raw_image):
    exif = raw_image.getexif()
    if not exif:
        return {}, (
            "No EXIF metadata found — common in AI-generated images, but also "
            "in screenshots, messaging-app re-saves, and edited photos."
        )
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    make = tags.get("Make")
    model_name = tags.get("Model")
    if make or model_name:
        label = f"{make or ''} {model_name or ''}".strip()
        return tags, (
            f"Camera metadata found ({label}) — suggests a real camera capture "
            "(though EXIF can be stripped or faked)."
        )
    return tags, "Some metadata found, but no camera make/model."


def compute_saliency(image):
    processor, model = load_saliency_model()
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    pixel_values.requires_grad_(True)

    out = model(pixel_values=pixel_values)
    artificial_idx = [k for k, v in model.config.id2label.items() if v.lower() == "artificial"][0]
    score = out.logits[0, artificial_idx]
    model.zero_grad()
    score.backward()

    saliency = pixel_values.grad.abs().sum(dim=1).squeeze().detach().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def overlay_heatmap(image, saliency):
    base = image.resize((saliency.shape[1], saliency.shape[0]))
    base_arr = np.array(base).astype(np.float32)
    heat = np.zeros_like(base_arr)
    heat[..., 0] = saliency * 255
    overlay = (base_arr * 0.6 + heat * 0.4).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    detectors = load_detectors()

    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(uploaded_file.name)

        raw_image = Image.open(uploaded_file)
        exif_tags, exif_note = check_exif(raw_image)
        image = raw_image.convert("RGB")

        with st.spinner("Analyzing..."):
            per_model_scores = []
            for name, detector in zip(MODEL_NAMES, detectors):
                score = artificial_score(detector(image))
                per_model_scores.append((name, score))
            ensemble_score = sum(s for _, s in per_model_scores) / len(per_model_scores)

            saliency = compute_saliency(image)
            heatmap_image = overlay_heatmap(image, saliency)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded image", width="stretch")
        with col2:
            st.image(heatmap_image, caption="Saliency heatmap (red = most influenced prediction)", width="stretch")

        verdict = "artificial" if ensemble_score > 0.5 else "real"
        st.write(f"**Ensemble verdict: {verdict}** ({ensemble_score:.1%} artificial-likelihood)")

        with st.expander("Per-model scores"):
            for name, score in per_model_scores:
                st.write(f"{name}: {score:.1%} artificial")

        st.info(exif_note)

        st.session_state.history.append({
            "file": uploaded_file.name,
            "verdict": verdict,
            "ensemble_artificial_score": round(ensemble_score, 3),
            **{name.split("/")[-1]: round(score, 3) for name, score in per_model_scores},
            "exif_present": bool(exif_tags),
        })

st.divider()
st.subheader("Session history")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), width="stretch")
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("No images analyzed yet this session.")
