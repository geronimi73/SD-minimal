import streamlit as st
import streamlit_ext as ste

import torch
import io
import numpy as np
import uuid
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
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter = T2IAdapter.from_pretrained(
        "Adapter/t2iadapter", 
        subfolder="sketch_sdxl_1.0", 
        torch_dtype=torch.float16, 
        adapter_type="full_adapter_xl")
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, 
        adapter=adapter, 
        vae=vae,
        scheduler=euler_a,
        torch_dtype=torch.float16, 
        variant="fp16", 
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    return pipe

def generate(params):
    pipe = load_pipe()
    image_out = pipe(
        **params
    ).images

    return image_out

def thumb(img):
    w,h=img.size
    img = img.resize((w//4,h//4))
    return img

## SIDEBAR
st.sidebar.markdown("## Settings")
prompt_addition = st.sidebar.text_input(
    "Prompt addition", 
    value="in real world, 4k photo, highly detailed")
negative_prompt = st.sidebar.text_input(
    "Negative prompt", 
    value="disfigured, extra digit, fewer digits, cropped, worst quality, low quality")
num_images = st.sidebar.slider(
    "Number of images to generate", min_value=1, max_value=10, value=1)
num_steps = st.sidebar.slider(
    "Number of steps", min_value=1, max_value=100, value=20)
guidance_scale = st.sidebar.slider(
    "Guidance scale", min_value=6, max_value=10, value=7)
adapter_conditioning_scale = st.sidebar.slider(
    "Adapter conditioning scale", min_value=0, max_value=100, value=90)
adapter_conditioning_factor = st.sidebar.slider(
    "Adapter conditioning factor", min_value=0, max_value=100, value=90)

col1, col2 = st.columns(2)

## MAIN LEFT
with col1:
    st.markdown("## Input")
    prompt = st.text_input("Prompt", value="a fluffy cat")
    thumb_placeholder = st.empty()
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

    generated_images=None
    if uploaded_file is not None:
        img_upload = Image.open(uploaded_file)
        thumb_placeholder.image(thumb(img_upload))

        from controlnet_aux.pidi import PidiNetDetector
        preprocessor = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
        img_preprocessed = preprocessor(
            img_upload, 
            detect_resolution=1024, 
            image_resolution=1024,
            apply_filter=True).convert("L")

        if st.button("Generate"):
            params=dict(
                image=img_preprocessed,
                num_inference_steps=num_steps,
                prompt=f"{prompt},{prompt_addition}" if len(prompt_addition.strip())>0 else prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                adapter_conditioning_scale=adapter_conditioning_scale/100,
                adapter_conditioning_factor=adapter_conditioning_factor/100,
                num_images_per_prompt=num_images
                )
            generated_images=generate(params)

## MAIN RIGHT
with col2:
    st.markdown("## Generated images")
    if generated_images is not None:
        for i in generated_images:
            st.image(i)
            buffer = io.BytesIO()
            i.save(buffer, format="PNG")
            buffer.seek(0) 
            ste.download_button("Download", data=buffer,mime="image/png", file_name="download.png")

            # st.download_button(label, i)
