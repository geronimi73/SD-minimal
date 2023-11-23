import streamlit as st
import torch
import io
import numpy as np
from diffusers.utils import load_image,numpy_to_pil
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from controlnet_aux import CannyDetector

st.set_page_config(layout="wide")

@st.cache_data
def load_pipe():
    from diffusers.utils import load_image, make_image_grid
    from diffusers import (
        T2IAdapter,
        StableDiffusionXLAdapterPipeline,
        DDPMScheduler,
        AutoencoderKL,
        EulerAncestralDiscreteScheduler
    )
    # git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    # git clone https://huggingface.co/Adapter/t2iadapter
    adapter = T2IAdapter.from_pretrained("Adapter/t2iadapter", subfolder="canny_sdxl_1.0", torch_dtype=torch.float16, adapter_type="full_adapter_xl")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, 
        adapter=adapter, 
        scheduler=scheduler,
        torch_dtype=torch.float16, 
        variant="fp16", 
    ).to("cuda")
    return pipe

def generate(num_steps, image_canny, positive_prompt, negative_prompt, guidance_scale, adapter_conditioning_scale, adapter_conditioning_factor):
    st.write(f"Loading model ...")
    pipe = load_pipe()

    st.write(f"Generating ...")
    image_out = pipe(
        prompt=positive_prompt,
        negative_prompt=f"{negative_prompt}, disfigured, extra digit, fewer digits, cropped, worst quality, low quality",
        image=image_canny,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        adapter_conditioning_scale=adapter_conditioning_scale,
        adapter_conditioning_factor=adapter_conditioning_factor
    ).images[0]

    return image_out

positive_prompt = st.sidebar.text_input("Prompt", value="a fluffy cat")
negative_prompt = st.sidebar.text_input("Negative Prompt", value="")
num_steps = st.sidebar.slider("Number of Steps", min_value=1, max_value=100, value=20)
guidance_scale = st.sidebar.slider("Guidance scale", min_value=6, max_value=10, value=7)
adapter_conditioning_scale = st.sidebar.slider("Adapter conditioning scale", min_value=0, max_value=100, value=90)
adapter_conditioning_factor = st.sidebar.slider("Adapter conditioning factor", min_value=0, max_value=100, value=50)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source image")
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)

        canny = CannyDetector()
        uploaded_image_canny = canny(uploaded_image, detect_resolution=512, image_resolution=1024).convert("L")

        with col1:
            st.image(uploaded_image)
            st.image(uploaded_image_canny)

with col2:
    st.subheader(f"Prompt: {positive_prompt}")

    st.subheader("Generated Image")
    image_placeholder = st.empty()

if st.sidebar.button("Generate"):
    if uploaded_image_canny is not None:
        generated_image=generate(num_steps, uploaded_image_canny, positive_prompt, negative_prompt, guidance_scale,adapter_conditioning_scale/100, adapter_conditioning_factor/100)
        image_placeholder.image(generated_image)
