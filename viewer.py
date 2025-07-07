import os
import torch
import streamlit as st
import torchvision.utils as vutils
from PIL import Image
from models.generator import Generator

# =============== CONFIGURATION ===============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 100
STYLE_DIM = 512
CHECKPOINT_PATH = "checkpoints/generator_epoch1.pth"  # Update path if needed

# =============== LOAD GENERATOR ===============
generator = Generator(latent_dim=LATENT_DIM, style_dim=STYLE_DIM).to(DEVICE)
generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
generator.eval()

# =============== SIDEBAR & UI ===============
st.set_page_config(page_title="Human Face Generator", layout="centered")
st.markdown(
    """
    <style>
    .stButton>button {
        font-size: 18px;
        padding: 0.5em 2em;
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #ff0000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Real Face Generation with StyleGan2 From Scratch")
st.markdown("Generate a gallery of realistic AI faces with  StyleGAN2 model!")

with st.sidebar:
    st.header("🔧 Settings")
    num_faces = st.slider("Number of Faces", min_value=1, max_value=500, value=20, step=5)
    seed = st.number_input("Random Seed (0=Random)", min_value=0, value=0, help="Use same seed to reproduce results.")

# =============== BUTTON LOGIC ===============
if st.button("🚀 Generate Faces"):
    # Set seed if specified
    if seed > 0:
        torch.manual_seed(seed)
    else:
        torch.seed()

    # Generate batch of faces
    noise = torch.randn(num_faces, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        fake_imgs = generator(noise).cpu()

    # Arrange images in a grid
    nrow = min(5, num_faces)  # 5 images per row
    grid = vutils.make_grid(fake_imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
    ndarr = grid.mul(255).byte().permute(1, 2, 0).numpy()
    img_pil = Image.fromarray(ndarr)

    st.image(img_pil, caption=f"{num_faces} Generated Faces", use_container_width=True)
