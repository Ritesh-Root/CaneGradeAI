<div align="center">

# 🎋 CaneGrade AI

### The "Net Clean Cane" Assessment System

**An AI-powered gate-entry system that stops sugar mills from paying for trash and water by calculating the Net Clean Cane (NCC) value of every truckload in real-time.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 🚨 The Problem

| Issue | Impact |
|---|---|
| Mills pay for **gross weight** | Farmers add trash (leaves/mud) and water to inflate weight |
| **No instant quality check** at the gate | Lab tests take 20+ minutes per truck |
| **Sugar inversion** over time | Cane harvested 48+ hours ago loses significant sugar content |
| **No penalty mechanism** | Mills absorb the financial loss silently |

> **Result:** Sugar mills lose ₹thousands per truck and ₹crores per season.

---

## 💡 The Solution — Hybrid AI Assessment

CaneGrade AI combines **Computer Vision** ("Eyes") with **Logistic Regression** ("Brain") to predict the true economic value of every truckload instantly.

### 👁️ The "Eyes" — Vision Module

| Detection | What It Finds | Impact |
|---|---|---|
| **Trash %** | Leaves, roots, mud | Reduces net weight |
| **Red Rot** | Disease / red discoloration | **Immediate rejection** |
| **Dry/Shrunken Skin** | Old, dehydrated cane | Lowers recovery rate |
| **Bounding Boxes** | Visual annotations on image | Operator can verify detections |

> *Currently simulated with randomized mock logic. Clear `TODO` comments mark where YOLOv8 `model.predict()` will plug in.*

### 🧠 The "Brain" — Regression Module

Trained on a synthetic dataset (800 samples) using **Scikit-Learn Linear Regression** with 5 features:

| Feature | Source |
|---|---|
| Trash % | Vision module |
| Time Lag (hours) | `current_time − harvest_time` |
| Cane Variety | User input (Co 0238, Co 86032, etc.) |
| Weather Conditions | User input (Rain, Heat, Clear) |
| Dry Skin Score | Vision module |

**Output:** `Predicted Sugar Recovery %`

---

## 🖥️ Live Demo Flow

```
📸 Upload Image  →  🔍 AI Detects Trash/Rot/DryS  →  🧮 Predicts Recovery %
                                                          ↓
   📊 Dashboard  ←  💰 Financial Impact (₹)  ←  🏷️ ACCEPT / WARNING / REJECT
```

### Output Dashboard Includes:

- ✅ **AI-Annotated Image** with bounding boxes (Trash, Mud, Rot, Dry Skin)
- 📊 **Core Metrics**: Trash %, Time Lag, Recovery %, Sugar Yield
- 💰 **Financial Impact**: Net Clean Weight, Loss Avoided (₹), Penalty Deduction
- 🏷️ **Recommendation**: ACCEPT (green) / WARNING + penalty % (yellow) / REJECT (red)
- 📝 Detailed breakdown table + reasoning explanation

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CaneGradeAI.git
cd CaneGradeAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run caneGradeAI_full.py
```

The app will open at **https://canegradeai.streamlit.app/** 🎉

---

## 📁 Project Structure

```
CaneGradeAI/
├── caneGradeAI_full.py   # 🎯 Main Streamlit application (single-file)
├── app.py                # Alias (same file)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
└── .gitignore            # Git ignore rules
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Streamlit** | Web UI & dashboard |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computations |
| **Scikit-Learn** | Linear Regression model |
| **Pillow (PIL)** | Image processing & bounding box rendering |
| **YOLOv8** *(planned)* | Real-time object detection for trash/rot |

---

## 🗺️ Roadmap

- [x] Simulated vision module with mock detections
- [x] Bounding box annotations on uploaded images
- [x] Red Rot & Dry Skin detection (simulated)
- [x] Weather-aware regression model
- [x] Financial impact calculator with ₹ values
- [x] Penalty deduction logic
- [ ] Train real YOLOv8 model on cane trash dataset
- [ ] Camera integration (live feed from gate)
- [ ] Database logging (per-truck history)
- [ ] Multi-language support (Hindi/English)
- [ ] SMS/WhatsApp alert to mill manager

---

## 🏆 Why This Project Wins

| Point | Reason |
|---|---|
| **Honest** | Doesn't claim to "see sugar" — uses vision for trash + logic for recovery |
| **Financial** | Saves mills money immediately at the weighbridge, not just "improving quality" |
| **Feasible** | YOLO can detect leaves vs cane in 24 hrs of training; the rest is Python math |
| **Scalable** | Single-file app → deploy on any server, add real model later |

---

## 👤 Author

**Ritesh Mahato**
📧 [riteshmahatowork@gmail.com](mailto:riteshmahatowork@gmail.com)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
