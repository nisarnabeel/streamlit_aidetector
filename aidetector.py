import pandas as pd
import streamlit as st
import torch
from PIL import ExifTags, Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.write("""
# AI-Generated Image Detector

Upload image
""")

st.caption(
    "⚠️ This is a rough, best-effort estimate from small open-source models, "
    "not a reliable verdict. It can be wrong, especially on newer generators, "
    "compressed/re-uploaded images, or heavily edited real photos."
)

MODEL_NAME = "haywoodsloan/ai-image-detector-deploy"


@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
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


def classify(image):
    processor, model = load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=0)
    return {model.config.id2label[i].lower(): probs[i].item() for i in range(len(probs))}


if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(uploaded_file.name)

        raw_image = Image.open(uploaded_file)
        exif_tags, exif_note = check_exif(raw_image)
        image = raw_image.convert("RGB")

        with st.spinner("Analyzing..."):
            probs = classify(image)

        st.image(image, caption="Uploaded image", width="stretch")

        artificial_score = probs.get("artificial", 0.0)
        verdict = "artificial" if artificial_score > 0.5 else "real"
        st.write(f"**Verdict: {verdict}**")

        st.subheader("Probability breakdown")
        for label, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{label}: {p:.1%}")
            st.progress(p)

        st.info(exif_note)

        st.session_state.history.append({
            "file": uploaded_file.name,
            "verdict": verdict,
            **{label: round(p, 3) for label, p in probs.items()},
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
