import streamlit as st
import torch
import io
import numpy as np
from diffusers.utils import load_image,numpy_to_pil
from PIL import Image
from streamlit_drawable_canvas import st_canvas

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
    adapter = T2IAdapter.from_pretrained("Adapter/t2iadapter", subfolder="sketch_sdxl_1.0", torch_dtype=torch.float16, adapter_type="full_adapter_xl")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, 
        adapter=adapter, 
        scheduler=scheduler,
        torch_dtype=torch.float16, 
        variant="fp16", 
    ).to("cuda")
    return pipe

def generate(num_steps, sketch_image, positive_prompt, negative_prompt, guidance_scale, adapter_conditioning_factor):
    st.write(f"Loading model...")
    pipe = load_pipe()
    sketch_image = numpy_to_pil(sketch_image)[0].convert("L")
    sketch_image = sketch_image.resize((1024, 1024))

    st.write(f"Generating for {num_steps} steps...")

    sketch_image_out = pipe(
        prompt=positive_prompt,
        negative_prompt=f"{negative_prompt}, disfigured, extra digit, fewer digits, cropped, worst quality, low quality",
        image=sketch_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        # adapter_conditioning_scale=0.9,
        adapter_conditioning_factor=adapter_conditioning_factor
    ).images[0]

    return sketch_image_out

st.sidebar.markdown("### Painting")

if 'paint' not in st.session_state:
    st.session_state.paint = True

def toggle_paint():
    st.session_state.paint = not st.session_state.paint
st.sidebar.toggle('Paint' if st.session_state.paint else "Erase", on_change=toggle_paint, value=st.session_state.paint)

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform")
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 8, 1)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)

st.sidebar.markdown("### Image generation")
positive_prompt = st.sidebar.text_input("Prompt", value="a fluffy cat")
negative_prompt = st.sidebar.text_input("Negative Prompt", value="cartoon")
num_steps = st.sidebar.slider("Number of Steps", min_value=1, max_value=50, value=10)
guidance_scale = st.sidebar.slider("Guidance scale", min_value=6, max_value=10, value=7)
adapter_conditioning_factor = st.sidebar.slider("Adapter conditioning factor", min_value=50, max_value=100, value=90)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sketch")
    canvas_result = st_canvas(
        fill_color="black",  
        stroke_width=stroke_width,
        stroke_color="white" if st.session_state.paint else "black", 
        background_color="black",
        height=500,
        width=500,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        drawing_mode=drawing_mode,
        key="canvas",
    )

with col2:
    st.subheader("Generated Image")
    image_placeholder = st.empty()

if st.sidebar.button("Generate"):
    generated_image=generate(num_steps, canvas_result.image_data, positive_prompt, negative_prompt, guidance_scale,adapter_conditioning_factor/100)
    image_placeholder.image(generated_image, width=500, caption='Generated Image')
