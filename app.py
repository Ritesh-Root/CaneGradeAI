"""
CaneGrade AI — Sugar Mill Net Clean Cane & Recovery Estimator
=============================================================
A hybrid Vision + Regression system that helps sugar mills estimate
the Net Clean Cane (NCC) and Sugar Recovery Rate of incoming truckloads.

Features:
 • Vision Module  — Trash %, Red Rot detection, Dry/Shrunken Skin, Bounding Boxes
 • Brain Module   — Weather-aware regression predicting Sugar Recovery %
 • Financial      — ₹ loss avoided, penalty deduction, NCC calculation

Run:  python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import io

# ──────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CaneGrade AI",
    page_icon="🎋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# Custom CSS — Professional Dark Theme
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ---------- Main App ---------- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #e0e0e0 !important;
    }

    /* ---------- Header ---------- */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00b09b, #96c93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #999;
        margin-top: 0;
    }

    /* ---------- Metric Cards ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30475e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetric"] label {
        color: #8899aa !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* ---------- Badge Helpers ---------- */
    .badge-accept {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: #fff; padding: 12px 28px; border-radius: 8px;
        font-weight: 700; font-size: 1.3rem; text-align: center;
        box-shadow: 0 4px 15px rgba(0,176,155,0.4);
    }
    .badge-warning {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #333; padding: 12px 28px; border-radius: 8px;
        font-weight: 700; font-size: 1.3rem; text-align: center;
        box-shadow: 0 4px 15px rgba(247,151,30,0.4);
    }
    .badge-reject {
        background: linear-gradient(135deg, #cb2d3e, #ef473a);
        color: #fff; padding: 12px 28px; border-radius: 8px;
        font-weight: 700; font-size: 1.3rem; text-align: center;
        box-shadow: 0 4px 15px rgba(203,45,62,0.4);
    }
    .badge-penalty {
        background: linear-gradient(135deg, #e65c00, #f9d423);
        color: #222; padding: 10px 24px; border-radius: 8px;
        font-weight: 700; font-size: 1.1rem; text-align: center;
        box-shadow: 0 4px 15px rgba(230,92,0,0.3);
        margin-top: 8px;
    }

    /* ---------- Financial Highlight ---------- */
    .loss-avoided {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: #fff; padding: 16px 24px; border-radius: 10px;
        font-weight: 800; font-size: 1.5rem; text-align: center;
        box-shadow: 0 6px 25px rgba(17,153,142,0.35);
        margin: 10px 0;
    }

    /* ---------- Dividers ---------- */
    .section-divider {
        border: none;
        border-top: 1px solid #30475e;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIETIES = ["Co 0238", "Co 86032", "CoJ 64", "CoS 8436", "Co 0118"]

WEATHER_OPTIONS = ["Clear / Sunny", "Cloudy", "Light Rain", "Heavy Rain", "Extreme Heat (>40°C)"]

# Cane price per ton (₹) — used for financial impact calculation
CANE_PRICE_PER_TON = 3500.0

THRESHOLDS = {
    "accept": 9.5,    # Recovery % ≥ 9.5 → ACCEPT
    "warning": 8.0,   # Recovery % ≥ 8.0 → WARNING + penalty deduction
    # Below 8.0 → REJECT
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  VISION MODULE  (Simulated)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_vision_analysis(image: Image.Image) -> dict:
    """
    Simulate the YOLOv8 / vision model analysis of a cane truck image.

    Returns a dict with:
        - trash_percent    : float  (2 – 15 %)
        - visual_purity    : float  (0 – 10)
        - red_rot_detected : bool   (Red-rot / disease flag)
        - dry_skin_score   : float  (0 – 10, higher = more shrunken/old)
        - bounding_boxes   : list   (simulated detection regions)

    ╔══════════════════════════════════════════════════════════╗
    ║  TODO: Replace with real YOLOv8 inference:              ║
    ║                                                         ║
    ║  from ultralytics import YOLO                           ║
    ║  model = YOLO("best.pt")                                ║
    ║  results = model.predict(source=image, conf=0.25)       ║
    ║                                                         ║
    ║  # Parse `results` to compute:                          ║
    ║  #   - trash_percent   from "leaf"/"mud" class boxes    ║
    ║  #   - red_rot_detected from "red_rot" class presence   ║
    ║  #   - dry_skin_score   from "dry_skin" class coverage  ║
    ║  #   - visual_purity    = f(trash, rot, dry)            ║
    ║  #   - bounding_boxes   = results[0].boxes.xyxy         ║
    ╚══════════════════════════════════════════════════════════╝
    """
    np.random.seed(hash(image.tobytes()[:64]) % (2**31))

    trash_percent = round(np.random.uniform(2.0, 15.0), 2)

    # Red Rot — ~15% chance of detection
    red_rot_detected = bool(np.random.random() < 0.15)

    # Dry / Shrunken Skin score (0 = fresh, 10 = very dried out)
    dry_skin_score = round(np.random.uniform(0.5, 6.0), 1)

    # Visual purity (composite)
    visual_purity = round(
        10.0
        - trash_percent * 0.40
        - (2.0 if red_rot_detected else 0.0)
        - dry_skin_score * 0.15
        + np.random.normal(0, 0.2),
        1,
    )
    visual_purity = float(np.clip(visual_purity, 0.0, 10.0))

    # ── Simulated bounding boxes (relative coords 0-1) ──
    w, h = image.size
    n_boxes = np.random.randint(2, 6)
    boxes = []
    labels = ["Trash/Leaves", "Mud", "Dry Skin", "Red Rot", "Root"]
    for i in range(n_boxes):
        x1 = np.random.uniform(0.05, 0.65)
        y1 = np.random.uniform(0.05, 0.65)
        bw = np.random.uniform(0.08, 0.30)
        bh = np.random.uniform(0.08, 0.25)
        label = labels[i % len(labels)]
        conf = round(np.random.uniform(0.55, 0.97), 2)
        boxes.append({
            "x1": int(x1 * w), "y1": int(y1 * h),
            "x2": int((x1 + bw) * w), "y2": int((y1 + bh) * h),
            "label": label, "confidence": conf,
        })

    return {
        "trash_percent": trash_percent,
        "visual_purity": visual_purity,
        "red_rot_detected": red_rot_detected,
        "dry_skin_score": dry_skin_score,
        "bounding_boxes": boxes,
    }


def draw_bounding_boxes(image: Image.Image, boxes: list) -> Image.Image:
    """Draw simulated detection bounding boxes on the image."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # Color map per class
    color_map = {
        "Trash/Leaves": "#FF4444",
        "Mud":          "#FF8800",
        "Dry Skin":     "#FFCC00",
        "Red Rot":      "#CC00FF",
        "Root":         "#00AAFF",
    }

    for box in boxes:
        color = color_map.get(box["label"], "#FFFFFF")
        # Rectangle
        draw.rectangle(
            [(box["x1"], box["y1"]), (box["x2"], box["y2"])],
            outline=color, width=3,
        )
        # Label background
        label_text = f'{box["label"]} {box["confidence"]:.0%}'
        text_bbox = draw.textbbox((box["x1"], box["y1"] - 16), label_text)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box["x1"], box["y1"] - 16), label_text, fill="white")

    return img


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  REGRESSION MODULE  (Synthetic Data + Linear Model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def build_regression_model():
    """
    Generate a synthetic training dataset and fit a Linear Regression
    model to predict Sugar Recovery Rate (%).

    Features:  Trash_Percent, Time_Lag_Hours, Variety (encoded),
               Weather_Code, Dry_Skin_Score
    Target:    Recovery_Rate (%)
    """
    rng = np.random.default_rng(42)
    n = 800

    varieties = rng.choice(VARIETIES, size=n)
    weather   = rng.choice(WEATHER_OPTIONS, size=n)
    trash     = rng.uniform(1.0, 18.0, size=n)
    time_lag  = rng.uniform(0.5, 72.0, size=n)
    dry_skin  = rng.uniform(0.0, 8.0, size=n)

    # Variety-specific base recovery rates
    variety_base = {
        "Co 0238":  11.0,
        "Co 86032": 10.5,
        "CoJ 64":   10.0,
        "CoS 8436":  9.8,
        "Co 0118":  10.2,
    }
    base = np.array([variety_base[v] for v in varieties])

    # Weather impact on recovery
    weather_penalty = {
        "Clear / Sunny":        0.0,
        "Cloudy":               0.0,
        "Light Rain":          -0.2,
        "Heavy Rain":          -0.5,
        "Extreme Heat (>40°C)":-0.4,
    }
    w_penalty = np.array([weather_penalty[w] for w in weather])

    # Recovery formula
    recovery = (
        base
        - 0.12 * trash            # trash penalty
        - 0.04 * time_lag         # inversion / staling penalty
        - 0.08 * dry_skin         # dry/old cane penalty
        + w_penalty               # weather penalty
        + rng.normal(0, 0.25, n)  # noise
    )
    recovery = np.clip(recovery, 3.0, 13.0)

    df = pd.DataFrame({
        "Trash_Percent":  trash,
        "Time_Lag_Hours": time_lag,
        "Variety":        varieties,
        "Weather":        weather,
        "Dry_Skin_Score": dry_skin,
        "Recovery_Rate":  recovery,
    })

    # Encode categoricals
    le_variety = LabelEncoder()
    le_weather = LabelEncoder()
    df["Variety_Enc"] = le_variety.fit_transform(df["Variety"])
    df["Weather_Enc"] = le_weather.fit_transform(df["Weather"])

    X = df[["Trash_Percent", "Time_Lag_Hours", "Variety_Enc",
            "Weather_Enc", "Dry_Skin_Score"]].values
    y = df["Recovery_Rate"].values

    model = LinearRegression()
    model.fit(X, y)

    return model, le_variety, le_weather, df


def predict_recovery(model, le_variety, le_weather,
                     trash_pct, time_lag_h, variety, weather, dry_skin) -> float:
    """Return predicted recovery % for given inputs."""
    v_enc = le_variety.transform([variety])[0]
    w_enc = le_weather.transform([weather])[0]
    X_new = np.array([[trash_pct, time_lag_h, v_enc, w_enc, dry_skin]])
    pred = model.predict(X_new)[0]
    return round(float(np.clip(pred, 0.0, 15.0)), 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  PENALTY & RECOMMENDATION LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_recommendation(recovery_pct, red_rot_detected, trash_pct):
    """
    Return (label, css_class, penalty_percent, reason).
    Penalty is a % deducted from payment.
    """
    # Immediate reject if Red Rot detected
    if red_rot_detected:
        return ("🚫  REJECT — Red Rot Detected", "badge-reject", 100.0,
                "Red rot disease detected. Load MUST be rejected to prevent contamination.")

    if recovery_pct >= THRESHOLDS["accept"] and trash_pct <= 5.0:
        return ("✅  ACCEPT", "badge-accept", 0.0,
                "Recovery and purity within acceptable range. No deductions.")

    if recovery_pct >= THRESHOLDS["warning"]:
        penalty = round(trash_pct * 0.6, 1)  # e.g. 10% trash → 6% deduction
        return (f"⚠️  WARNING — Deduct {penalty}% from Payment", "badge-warning", penalty,
                f"High trash ({trash_pct:.1f}%). Apply {penalty}% penalty deduction.")

    penalty = round(trash_pct * 0.8 + 2.0, 1)
    return (f"🚫  REJECT — Recovery too low ({recovery_pct:.1f}%)", "badge-reject", penalty,
            f"Recovery below {THRESHOLDS['warning']}%. Recommend rejection or heavy penalty ({penalty}%).")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  STREAMLIT  UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # Load model once (cached)
    model, le_variety, le_weather, train_df = build_regression_model()

    # ───── Header ─────
    st.markdown('<p class="hero-title">🎋 CaneGrade AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Hybrid Vision + Regression system for Net Clean Cane &amp; Sugar Recovery estimation</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ───── Sidebar Inputs ─────
    with st.sidebar:
        st.header("📥  Input Panel")
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Upload Cane Image", type=["jpg", "jpeg", "png"],
            help="Photo of the cane truck or pile",
        )

        st.markdown("---")
        st.subheader("📋 Truck Details")

        harvest_time = st.date_input("Harvest Date", value=datetime.now().date())
        harvest_hour = st.time_input("Harvest Time", value=datetime.now().time())
        harvest_dt = datetime.combine(harvest_time, harvest_hour)

        gross_weight = st.number_input(
            "Gross Weight (Tons)", min_value=1.0, max_value=100.0, value=25.0, step=0.5,
        )

        variety = st.selectbox("Cane Variety", options=VARIETIES)

        st.markdown("---")
        st.subheader("🌦️ Weather at Harvest")
        weather = st.selectbox("Weather Conditions", options=WEATHER_OPTIONS)
        temperature = st.slider("Temperature (°C)", min_value=10, max_value=50, value=32)

        st.markdown("---")
        analyze_btn = st.button("🚀  Analyze Truck", use_container_width=True, type="primary")

    # ───── Main Content ─────
    if not analyze_btn:
        # Landing state
        st.info("👈  Fill in the details in the sidebar and click **Analyze Truck** to begin.")

        with st.expander("📊  Peek at Synthetic Training Data"):
            st.dataframe(
                train_df.drop(columns=["Variety_Enc", "Weather_Enc"]).head(20).style.format({
                    "Trash_Percent": "{:.1f}%",
                    "Time_Lag_Hours": "{:.1f} h",
                    "Dry_Skin_Score": "{:.1f}",
                    "Recovery_Rate": "{:.2f}%",
                }),
                use_container_width=True,
            )
        return

    # ── Validate Image ──
    if uploaded_file is None:
        st.warning("⚠️  Please upload an image of the cane truck to proceed.")
        return

    image = Image.open(uploaded_file)

    # ── Compute time lag ──
    now = datetime.now()
    time_lag_hours = max((now - harvest_dt).total_seconds() / 3600.0, 0.0)
    time_lag_hours = round(time_lag_hours, 1)

    # ── Vision analysis ──
    vision = simulate_vision_analysis(image)
    trash_pct       = vision["trash_percent"]
    visual_purity   = vision["visual_purity"]
    red_rot         = vision["red_rot_detected"]
    dry_skin        = vision["dry_skin_score"]
    bboxes          = vision["bounding_boxes"]

    # ── Draw bounding boxes on image ──
    annotated_image = draw_bounding_boxes(image, bboxes)

    # ── Recovery prediction ──
    recovery_pct = predict_recovery(
        model, le_variety, le_weather,
        trash_pct, time_lag_hours, variety, weather, dry_skin,
    )

    # ── Financial calculations ──
    trash_weight     = round(gross_weight * (trash_pct / 100.0), 2)
    net_clean_weight = round(gross_weight - trash_weight, 2)
    estimated_sugar  = round(net_clean_weight * (recovery_pct / 100.0), 2)

    # ── Recommendation + penalty ──
    rec_label, rec_class, penalty_pct, rec_reason = compute_recommendation(
        recovery_pct, red_rot, trash_pct,
    )
    penalty_amount = round(gross_weight * CANE_PRICE_PER_TON * (penalty_pct / 100.0), 0)
    loss_avoided   = round(trash_weight * CANE_PRICE_PER_TON, 0)

    # ════════════════════════════════════════
    #   DASHBOARD LAYOUT
    # ════════════════════════════════════════

    # ── Row 1: Annotated Image + Recommendation ──
    col_img, col_rec = st.columns([2, 1])

    with col_img:
        st.subheader("📸  AI-Annotated Image")
        st.image(annotated_image, use_container_width=True,
                 caption=f"Variety: {variety}  |  Detections: {len(bboxes)}")

    with col_rec:
        st.subheader("🏷️  Recommendation")
        st.markdown(f'<div class="{rec_class}">{rec_label}</div>', unsafe_allow_html=True)

        if penalty_pct > 0 and penalty_pct < 100:
            st.markdown(
                f'<div class="badge-penalty">💸 Penalty Deduction: {penalty_pct}% '
                f'(₹{penalty_amount:,.0f})</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.metric("Visual Purity Score", f"{visual_purity} / 10")
        st.metric("Cane Variety", variety)

        if red_rot:
            st.error("🔴 **RED ROT DETECTED** — Load must be rejected!")
        if dry_skin >= 4.0:
            st.warning(f"⚠️ **Dry/Shrunken Skin** score: {dry_skin}/10 — Cane may be old.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Row 2: Vision Detections Summary ──
    st.subheader("�  Vision Detections")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("🗑️ Trash %", f"{trash_pct:.1f}%")
    d2.metric("🔴 Red Rot", "DETECTED" if red_rot else "Clear")
    d3.metric("🥀 Dry Skin Score", f"{dry_skin:.1f} / 10")
    d4.metric("👁️ Visual Purity", f"{visual_purity:.1f} / 10")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Row 3: Core Metrics ──
    st.subheader("📊  Core Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱️ Time Lag", f"{time_lag_hours:.1f} hrs")
    m2.metric("🌦️ Weather", weather)
    m3.metric("🧪 Predicted Recovery", f"{recovery_pct:.2f}%")
    m4.metric("🍬 Est. Sugar Yield", f"{estimated_sugar:.2f} T")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Row 4: Financial Impact ──
    st.subheader("💰  Financial Impact")

    # Highlight: Loss Avoided
    st.markdown(
        f'<div class="loss-avoided">'
        f'🛡️ Estimated Loss Avoided: ₹{loss_avoided:,.0f} on this truck'
        f'</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Gross Weight", f"{gross_weight:.1f} T")
    f2.metric("Trash Weight", f"{trash_weight:.2f} T",
              delta=f"-{trash_weight:.2f} T", delta_color="inverse")
    f3.metric("Net Clean Weight", f"{net_clean_weight:.2f} T")
    f4.metric("Penalty Deduction", f"₹{penalty_amount:,.0f}" if penalty_pct > 0 else "₹0")

    # ── Comparison bar chart ──
    chart_df = pd.DataFrame({
        "Category": ["Gross Weight", "Net Clean Weight", "Trash Weight"],
        "Tons": [gross_weight, net_clean_weight, trash_weight],
    }).set_index("Category")
    st.bar_chart(chart_df, color="#00b09b")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Row 5: Detailed Breakdown ──
    with st.expander("📋  Detailed Breakdown", expanded=False):
        detail = {
            "Parameter": [
                "Cane Variety",
                "Gross Weight (T)",
                "Detected Trash (%)",
                "Red Rot Detected",
                "Dry Skin Score",
                "Trash Weight (T)",
                "Net Clean Weight (T)",
                "Time Lag (Hours)",
                "Weather Condition",
                "Visual Purity Score",
                "Predicted Recovery (%)",
                "Estimated Sugar Yield (T)",
                "Penalty Deduction (%)",
                "Penalty Amount (₹)",
                "Loss Avoided (₹)",
                "Recommendation",
            ],
            "Value": [
                variety,
                f"{gross_weight:.1f}",
                f"{trash_pct:.2f}",
                "YES ⛔" if red_rot else "No ✅",
                f"{dry_skin:.1f} / 10",
                f"{trash_weight:.2f}",
                f"{net_clean_weight:.2f}",
                f"{time_lag_hours:.1f}",
                weather,
                f"{visual_purity}",
                f"{recovery_pct:.2f}",
                f"{estimated_sugar:.2f}",
                f"{penalty_pct:.1f}%",
                f"₹{penalty_amount:,.0f}",
                f"₹{loss_avoided:,.0f}",
                rec_label,
            ],
        }
        st.table(pd.DataFrame(detail))

    # ── Reason / Recommendation explanation ──
    with st.expander("📝  Recommendation Reasoning"):
        st.write(rec_reason)
        if time_lag_hours > 24:
            st.warning(
                f"⏳ **Staling Alert**: Cane harvested {time_lag_hours:.0f} hours ago. "
                f"Sugar inversion increases significantly beyond 24 hours."
            )
        if weather in ["Heavy Rain", "Extreme Heat (>40°C)"]:
            st.info(
                f"🌦️ **Weather Factor**: '{weather}' conditions reduce recovery potential. "
                f"This has been factored into the prediction."
            )

    # ── Footer ──
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:right; color:#666; font-size:0.78rem; padding:4px 12px;">'
        'Created by <strong>Ritesh Mahato</strong> · '
        '<a href="mailto:riteshmahatowork@gmail.com" style="color:#00b09b;">'
        'riteshmahatowork@gmail.com</a>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "CaneGrade AI v2.0 — Hybrid Vision + Regression Model  •  "
        "Trash • Red Rot • Dry Skin • Weather-Aware  •  "
        "Built with Streamlit & Scikit-Learn"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    main()
