import os
import json
import joblib
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from models import DermoViT
from interpretability import GradCAMPlusPlus, compute_attention_rollout

# --- Page Config & UI Setup ---
st.set_page_config(page_title="DermoViT Explorer", layout="wide", page_icon="🔬")
st.title("🔬 DermoViT Native Explorer")
st.markdown("""
Upload a dermoscopic image and adjust patient demographics to trigger the advanced dual-stream architecture. 
The system outputs **not just the probability of cancer**, but the mathematical **localization heatmaps** dictating the decision.
""")

# --- Config & Defaults ---
META_DIM = 30
IMG_SIZE = 224
MALIGNANT_THRESHOLD = 0.0545  # STEP 3: calibrated threshold for alpha=0.25 Focal Loss training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    ckpt_path = 'dermovit_best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'checkpoints/dermovit_best.pth'
        
    meta_dim = META_DIM
    ckpt = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'meta_dim' in ckpt:
            meta_dim = ckpt['meta_dim']

    model = DermoViT(meta_dim=meta_dim, drop_rate=0.0, freeze_backbone=False)
    
    if ckpt is not None:
        st.sidebar.success("✅ Loaded trained DermoViT weights!")
        try:
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
        except Exception as e:
            st.sidebar.error(f"Failed to load weights: {e}")
    else:
        st.sidebar.warning("⚠️ Running in Demo Mode (Untrained Weights)")
        st.sidebar.info("The logic and internal heatmaps will run perfectly to demonstrate the architecture, but the final score will be random until weights are provided.")
    
    model.to(device)
    model.eval()
    return model

# STEP 2: load the scaler and column order saved during training
@st.cache_resource
def load_scaler():
    scaler_path = 'checkpoints/meta_scaler.pkl'
    cols_path   = 'checkpoints/meta_cols.json'
    if os.path.exists(scaler_path) and os.path.exists(cols_path):
        return joblib.load(scaler_path), json.load(open(cols_path))
    return None, None

model = load_model()
scaler, meta_cols = load_scaler()

META_DIM_ACTUAL = model.film.meta_dim

# --- Sidebar Inputs ---
st.sidebar.header("Patient Clinical Record")

age      = st.sidebar.slider("Age", 0, 100, 45)
sex      = st.sidebar.selectbox("Sex", ["Male", "Female"])
anatomy  = st.sidebar.selectbox("Anatomical Site", ["Head/Neck", "Torso", "Extremity"])
symm_score = st.sidebar.slider("Clinical Asymmetry Score", 0.0, 1.0, 0.5)

st.sidebar.markdown("*(Note: For this interactive demo, the remaining tabular features are mocked at zero-mean)*")

# STEP 2: build metadata vector using saved column order + scaler
meta_raw = np.zeros((1, META_DIM_ACTUAL), dtype=np.float32)

if meta_cols is not None:
    col_idx = {c: i for i, c in enumerate(meta_cols)}

    if 'age_approx' in col_idx:
        meta_raw[0, col_idx['age_approx']] = float(age)

    if 'sex_male' in col_idx:
        meta_raw[0, col_idx['sex_male']] = 1.0 if sex == "Male" else 0.0

    site_map = {
        'Head/Neck': 'anatom_site_general_head/neck',
        'Torso':     'anatom_site_general_torso',
        'Extremity': 'anatom_site_general_lower extremity',
    }
    site_col = site_map.get(anatomy)
    if site_col and site_col in col_idx:
        meta_raw[0, col_idx[site_col]] = 1.0

    if scaler is not None:
        # scaler expects all 12 FILM_NUMERIC columns in the original order
        FILM_NUMERIC = [
            'age_approx', 'clin_size_long_diam_mm',
            'tbp_lv_symm_2axis', 'tbp_lv_eccentricity', 'tbp_lv_norm_border',
            'tbp_lv_norm_color', 'tbp_lv_color_std_mean', 'tbp_lv_radial_color_std_max',
            'tbp_lv_nevi_confidence', 'tbp_lv_dnn_lesion_confidence',
            'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio',
        ]
        # build a 1×12 array (zeros = dataset median after imputation)
        numeric_input = np.zeros((1, len(FILM_NUMERIC)), dtype=np.float32)
        numeric_input[0, 0] = float(age)   # age_approx is index 0

        scaled = scaler.transform(numeric_input)   # now receives all 12 → no error

        # write scaled values back into the correct positions in meta_raw
        for i, col in enumerate(FILM_NUMERIC):
            if col in col_idx:
                meta_raw[0, col_idx[col]] = scaled[0, i]
    else:
        # scaler not found: fall back to manual normalisation for age only
        if 'age_approx' in col_idx:
            meta_raw[0, col_idx['age_approx']] = (age - 50) / 20.0
else:
    # meta_cols not found: original fallback so app still runs in demo mode
    meta_raw[0, 0] = (age - 50) / 20.0
    meta_raw[0, 1] = 1.0 if sex == "Male" else -1.0
    meta_raw[0, 2] = symm_score

meta_vec = meta_raw

uploaded_file = st.sidebar.file_uploader("Upload Dermoscopy Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_t   = val_transform(image_pil).unsqueeze(0).to(device)

    resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
    image_np = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE), resample_filter)) / 255.0

    st.markdown("### 1. Neural Network Prediction")
    meta_t = torch.tensor(meta_vec).to(device)

    with torch.no_grad():
        logit, acag_attn = model(image_t, meta_t)
        prob = torch.sigmoid(logit).item()

    st.metric(label="Malignancy Probability", value=f"{prob*100:.2f}%")
    st.caption(f"Model range: 2–40% | Threshold: {MALIGNANT_THRESHOLD*100:.1f}% | "
               f"{'⚠️ Above threshold' if prob > MALIGNANT_THRESHOLD else '✅ Below threshold'}")

    if prob > MALIGNANT_THRESHOLD:
        st.error("🚨 HIGH RISK: Urgent Biopsy Recommended")
    else:
        st.success("✅ LOW RISK: Benign (Routine Monitor)")

    # STEP 3: use calibrated threshold instead of 0.5
    

    st.markdown("---")
    st.markdown("### 2. Explainable AI (XAI) Heatmaps")
    st.markdown("The model identifies **where** the problem is using three distinct mathematical techniques:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(image_np, caption="Original uploaded image", use_container_width=True)
        st.markdown("**Original Image**")

    with col2:
        with st.spinner("Computing CNN Grad-CAM++..."):
            target_layer = model.cnn_backbone
            grad_cam     = GradCAMPlusPlus(model, target_layer)
            with torch.enable_grad():
                cam_map = grad_cam(image_t, meta_t)

            fig, ax = plt.subplots()
            ax.imshow(image_np)
            ax.imshow(cam_map, cmap='jet', alpha=0.5)
            ax.axis('off')
            st.pyplot(fig)
            st.markdown("**Grad-CAM++**\nLocates irregular borders via the local EfficientNet stream.")

    with col3:
        with st.spinner("Computing ViT Rollout..."):
            rollout_map = compute_attention_rollout(model.vit_backbone, image_t)
            rollout_up  = np.array(Image.fromarray(
                (rollout_map * 255).astype(np.uint8)
            ).resize((IMG_SIZE, IMG_SIZE))) / 255.0

            fig2, ax2 = plt.subplots()
            ax2.imshow(image_np)
            ax2.imshow(rollout_up, cmap='inferno', alpha=0.5)
            ax2.axis('off')
            st.pyplot(fig2)
            st.markdown("**Attention Rollout**\nLocates macroscopic shape context via the global DeiT stream.")

    with col4:
        with st.spinner("Computing ACAG..."):
            acag_avg = acag_attn.mean(dim=[0,1,3]).reshape(7,7).detach().cpu().numpy()
            acag_avg = (acag_avg - acag_avg.min()) / (acag_avg.max() - acag_avg.min() + 1e-8)
            acag_up  = np.array(Image.fromarray(
                (acag_avg * 255).astype(np.uint8)
            ).resize((IMG_SIZE, IMG_SIZE))) / 255.0

            fig3, ax3 = plt.subplots()
            ax3.imshow(image_np)
            ax3.imshow(acag_up, cmap='plasma', alpha=0.6)
            ax3.axis('off')
            st.pyplot(fig3)
            st.markdown("**ACAG Asymmetry Map**\nNovel bias gate explicitly calculating and locating mathematical asymmetry.")