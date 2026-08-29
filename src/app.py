import streamlit as st
from PIL import Image
import tempfile
from pathlib import Path

from src.inference.predict import load_model
from src.vision.feature_extractor import ResNetFeatureExtractor


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
)


# --------------------------------------------------
# Load model and feature extractor
# --------------------------------------------------

@st.cache_resource
def load_captioning_pipeline():
    model, vocabulary, device = load_model()

    feature_extractor = ResNetFeatureExtractor(
        device=str(device)
    )

    return model, vocabulary, feature_extractor


# --------------------------------------------------
# User Interface
# --------------------------------------------------

st.title("🖼️ Image Caption Generator")

st.write(
    "Upload an image and the trained model will "
    "generate a caption describing it."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True,
    )

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            try:
                # Load model and feature extractor
                model, vocabulary, feature_extractor = (
                    load_captioning_pipeline()
                )

                # Save uploaded image temporarily
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False,
                ) as temp_file:

                    image.save(temp_file.name)
                    temp_image_path = Path(temp_file.name)

                # Extract ResNet50 features
                image_features = feature_extractor.extract(
                    temp_image_path
                )

                # Generate caption
                caption = model.generate_caption(
                    image_features,
                    vocabulary,
                )

                # Remove temporary image
                temp_image_path.unlink(
                    missing_ok=True
                )

                # Display result
                st.success("Caption generated!")

                st.subheader("Generated Caption")

                st.write(
                    f"**{caption}**"
                )

            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )
