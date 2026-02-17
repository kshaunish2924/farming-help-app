# app.py — FULL FILE (Your existing Farming Help features + NDVI route + Offline LLM Chatbot via Ollama + Quick Login)
# ✅ Keeps your existing routes and logic
# ✅ Adds /login + /logout (session-based)
# ✅ Protects "/" (main dashboard) behind login_required
# ✅ Chatbot works offline via Ollama: /chat and /chat/reset
# ✅ NDVI route stays separate: /ndvi/analyze
#
# NOTE:
# - You MUST create templates/login.html (I can paste it again if you want).
# - Your templates/index.html should include the chatbot JS (the one I gave earlier).

import os
import time
import json
import re
import traceback
from typing import List, Optional, Tuple, Dict
from functools import wraps

import numpy as np

# Pillow is required for Grad-CAM and NDVI utils.
try:
    from PIL import Image
except Exception:
    Image = None

import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, flash
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Optional: load .env for local dev
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except Exception:
    pass

# Optional NDVI utilities (your ndvi_utils.py)
try:
    from ndvi_utils import load_rgb_image, ndvi_metrics
except Exception:
    load_rgb_image = None
    ndvi_metrics = None


# --------------------------------
# Flask setup & constants
# --------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------------------
# QUICK LOGIN CONFIG (demo-friendly)
# --------------------------------
APP_USER = os.environ.get("APP_USER", "admin")
APP_PASS = os.environ.get("APP_PASS", "admin123")


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    # If already logged in, go to dashboard
    if session.get("logged_in"):
        return redirect(url_for("index"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        if username == APP_USER and password == APP_PASS:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))
        else:
            flash("Invalid username/password", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --------------------------------
# Weather config
# --------------------------------
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
OWM_CURRENT = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"
WEATHER_TTL_SEC = 600  # 10 minutes cache


# --------------------------------
# Model config
# --------------------------------
MODEL_PATH = "plant_disease_model.keras"
IMG_SIZE = 224  # MobileNetV2 default


# --------------------------------
# Load model once (safe)
# --------------------------------
model = None
BASE_CNN = None

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Try to grab the internal MobileNetV2 submodel
    try:
        BASE_CNN = model.get_layer("mobilenetv2_1.00_224")
    except Exception:
        BASE_CNN = None
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.Model):
                BASE_CNN = lyr
                break
        if BASE_CNN is None:
            BASE_CNN = model

except Exception as e:
    print("❌ Model load failed:", e)
    model = None
    BASE_CNN = None


# --------------------------------
# Load class names
# --------------------------------
CLASS_NAMES: Optional[List[str]] = None
if os.path.exists("classes.json"):
    try:
        with open("classes.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                CLASS_NAMES = data
            elif isinstance(data, dict):
                CLASS_NAMES = [data[str(i)] for i in range(len(data))]
    except Exception:
        CLASS_NAMES = None


# --------------------------------
# Utils: images & prediction
# --------------------------------
def preprocess_image(img_path: str) -> np.ndarray:
    img = kimage.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = kimage.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_label_multiclass(img_path: str) -> str:
    if model is None:
        return "MODEL_NOT_LOADED"
    arr = preprocess_image(img_path)
    pred = model.predict(arr, verbose=0)
    idx = int(np.argmax(pred[0]))
    if CLASS_NAMES:
        return CLASS_NAMES[idx]
    return f"class_{idx}"


# --------------------------------
# Load disease rules JSON
# --------------------------------
DISEASE_RULES: Dict = {}
try:
    with open("disease_rules.json", "r", encoding="utf-8") as f:
        DISEASE_RULES = json.load(f)
except Exception as e:
    print("Could not load disease_rules.json:", e)


# --------------------------------
# Weather-based risk score
# --------------------------------
def compute_risk_level(humidity=None, temperature=None) -> str:
    if humidity is None:
        return "Unknown"

    if humidity < 60:
        score = 1
    elif humidity < 75:
        score = 2
    elif humidity < 85:
        score = 3
    else:
        score = 4

    if temperature is not None and 22 <= temperature <= 30:
        score = min(score + 1, 4)

    return {1: "Low", 2: "Moderate", 3: "High", 4: "Severe"}.get(score, "Unknown")


# --------------------------------
# Severity mapping
# --------------------------------
def determine_severity(ndvi_delta=None) -> str:
    if ndvi_delta is None:
        return "moderate"
    if ndvi_delta < 0.1:
        return "mild"
    elif ndvi_delta < 0.25:
        return "moderate"
    else:
        return "severe"


# --------------------------------
# Advanced Recommendation Engine
# --------------------------------
def get_advanced_recommendations(label: Optional[str],
                                 ndvi_delta=None,
                                 humidity=None,
                                 temperature=None) -> List[str]:
    if not label:
        return ["No disease label detected."]

    key = label.lower().replace(" ", "_")
    key = re.sub(r"_+", "_", key)

    # HEALTHY CASE
    if "healthy" in key:
        matched = None
        for disease_key in DISEASE_RULES.keys():
            if "healthy" in disease_key and disease_key in key:
                matched = disease_key
                break

        if not matched:
            return ["Healthy plant detected — no rules found."]

        rules = DISEASE_RULES[matched]
        if "severity" not in rules or "normal" not in rules["severity"]:
            return ["Healthy plant detected — rules incomplete."]

        data = rules["severity"]["normal"]
        recs: List[str] = []
        recs.append("### Healthy Plant Maintenance:")
        recs.extend(f"- {x}" for x in data.get("actions", []))

        recs.append("\n### Organic Care:")
        recs.extend(f"- {x}" for x in data.get("organic", []))

        recs.append("\n### Preventive Notes:")
        recs.extend(f"- {x}" for x in data.get("chemical", []))

        if "environment_triggers" in rules:
            trig = rules["environment_triggers"]
            if humidity is not None and humidity > trig.get("humidity_gt", 999):
                recs.append("\n⚠ **Warning:** " + trig.get("warning", ""))
            if temperature is not None and "temp_range" in trig:
                lo, hi = trig["temp_range"]
                if lo <= temperature <= hi:
                    recs.append("⚠ **Temperature may increase disease risk — monitor regularly.**")

        return recs

    # DISEASE CASE
    matched = None
    for disease in DISEASE_RULES.keys():
        if disease in key:
            matched = disease
            break

    if not matched:
        return ["Disease detected, but no detailed rules available."]

    rules = DISEASE_RULES[matched]
    severity = determine_severity(ndvi_delta)

    if "severity" not in rules or severity not in rules["severity"]:
        return ["Severity level not found in rules."]

    data = rules["severity"][severity]
    recs: List[str] = []

    recs.append(f"### Disease Severity: {severity.capitalize()}")
    recs.append("")
    recs.append("### Chemical Treatment:")
    recs.extend(f"- {x}" for x in data.get("chemical", []))

    recs.append("\n### Organic Alternatives:")
    recs.extend(f"- {x}" for x in data.get("organic", []))

    recs.append("\n### Immediate Actions:")
    recs.extend(f"- {x}" for x in data.get("actions", []))

    if "environment_triggers" in rules:
        trig = rules["environment_triggers"]
        if humidity is not None and humidity > trig.get("humidity_gt", 999):
            recs.append("\n⚠ **Warning:** " + trig.get("warning", ""))
        if temperature is not None and "temp_range" in trig:
            lo, hi = trig["temp_range"]
            if lo <= temperature <= hi:
                recs.append("⚠ **Temperature range is ideal for rapid spread.**")

    return recs


# --------------------------------
# Marketplace & Laws JSON loaders
# --------------------------------
def _load_json_file(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


PRODUCTS = _load_json_file("products.json") or []
LAWS = _load_json_file("laws.json") or {}


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def infer_crop_from_label(predicted_label: Optional[str]) -> Optional[str]:
    if not predicted_label:
        return None
    parts = re.split(r"___|__|:|/|\\|-|,|\s+", predicted_label)
    first = (parts[0] if parts else predicted_label).strip().lower()
    known = {"tomato", "potato", "wheat", "rice", "chili", "maize",
             "apple", "grape", "citrus", "banana", "cotton", "soybean", "mustard", "pepper", "bell"}
    return first if first in known else None


def marketplace_recommendations(predicted_label: Optional[str]) -> List[dict]:
    if not PRODUCTS:
        return []
    label = _norm(predicted_label)
    tags = [k for k in ["blight", "rust", "mildew", "rot", "spot", "scab", "bacterial", "insect", "mite"] if k in label]
    if not tags:
        tags = ["general"]

    crop = infer_crop_from_label(predicted_label)
    out = []
    for p in PRODUCTS:
        ftags = [_norm(t) for t in p.get("for_tags", [])]
        crops = [_norm(c) for c in p.get("crops", [])]
        if any(t in ftags for t in tags) and (crop is None or crop in crops or not crops):
            out.append(p)

    final = []
    seen = set()
    for p in out:
        pid = p.get("id") or p.get("name")
        if pid in seen:
            continue
        seen.add(pid)
        final.append(p)
        if len(final) >= 6:
            break
    return final


def laws_for_state(state: Optional[str], predicted_label: Optional[str]) -> List[dict]:
    if not state:
        return []
    state_data = LAWS.get(state, {})
    items = list(state_data.get("general", []))
    crop = infer_crop_from_label(predicted_label)
    if crop and crop in state_data:
        items += state_data.get(crop, [])
    return items


# --------------------------------
# Soil Add-on
# --------------------------------
SOIL_CATALOG = _load_json_file("soil/soil_catalog.json") or {}
STATE_SOILS = _load_json_file("soil/state_soils.json") or {}


def get_state_default_soils(state: str) -> List[str]:
    st = (state or "").strip()
    if not st or st not in STATE_SOILS:
        return []
    return STATE_SOILS[st].get("default_soils", []) or []


def choose_soil(state: str, manual_soil: Optional[str]) -> Tuple[Optional[str], Optional[dict], str]:
    manual_soil = (manual_soil or "").strip()
    if manual_soil and manual_soil in SOIL_CATALOG:
        return manual_soil, SOIL_CATALOG[manual_soil], "manual"

    defaults = get_state_default_soils(state)
    if defaults:
        s = defaults[0]
        return s, SOIL_CATALOG.get(s), "default"

    return None, None, "none"


def soil_weather_leaf_fusion_notes(soil_info: Optional[dict],
                                  weather: Optional[dict],
                                  predicted_label: Optional[str]) -> List[str]:
    notes = []
    if not soil_info:
        return notes

    traits = (soil_info.get("traits") or {})
    base = (soil_info.get("baseline") or {})

    notes.append("### Soil-Based Advisory:")
    for x in base.get("amendments", []):
        notes.append(f"- **Soil amendment:** {x}")
    for x in base.get("irrigation", []):
        notes.append(f"- **Irrigation:** {x}")
    for x in base.get("warnings", []):
        notes.append(f"- ⚠ **Soil warning:** {x}")

    if weather:
        hum = weather.get("humidity")
        temp = weather.get("temp_c")

        if hum is not None and hum >= 75 and traits.get("drainage") in ("poor", "poor_to_medium"):
            notes.append("- ⚠ **High humidity + poor drainage**: Increase drainage, avoid over-irrigation, keep foliage dry at night.")

        if temp is not None and 22 <= float(temp) <= 30 and traits.get("water_retention") in ("high",):
            notes.append("- ⚠ **Warm + high water retention soil**: Fungal spread risk increases; ensure spacing, airflow, and morning watering only.")

    label = (predicted_label or "").lower()
    if label:
        if any(k in label for k in ["blight", "spot", "mildew", "rust"]):
            if traits.get("drainage") in ("poor", "poor_to_medium"):
                notes.append("- ✅ Disease looks fungal + soil drains poorly: prioritize drainage + avoid wet leaves; spray timing matters (avoid before rain).")
        if "healthy" in label:
            notes.append("- ✅ Plant looks healthy: follow soil maintenance + preventive schedule (compost + proper irrigation).")

    return notes


# --------------------------------
# Weather cache + functions
# --------------------------------
_weather_cache: Dict[str, Tuple[float, Optional[dict], Optional[str]]] = {}
_forecast_cache: Dict[str, Tuple[float, Optional[List[dict]], Optional[str]]] = {}


def _city_key(city: str) -> str:
    return city.strip().lower()


def fetch_weather(city_name: str) -> Tuple[Optional[dict], Optional[str]]:
    if not city_name:
        return None, "Enter a city for weather."
    if not OPENWEATHER_API_KEY:
        return None, "OPENWEATHER_API_KEY is not set."

    key = _city_key(city_name)
    now = time.time()

    if key in _weather_cache and now - _weather_cache[key][0] < WEATHER_TTL_SEC:
        return _weather_cache[key][1], _weather_cache[key][2]

    try:
        params = {"q": city_name, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        r = requests.get(OWM_CURRENT, params=params, timeout=8)

        if r.status_code == 404 and "," not in city_name:
            params["q"] = f"{city_name},IN"
            r = requests.get(OWM_CURRENT, params=params, timeout=8)

        if r.status_code != 200:
            try:
                data = r.json()
                msg = data.get("message", f"HTTP {r.status_code}")
            except Exception:
                msg = f"HTTP {r.status_code}"
            _weather_cache[key] = (now, None, f"Weather error: {msg}")
            return None, f"Weather error: {msg}"

        data = r.json()
        result = {
            "city": f"{data.get('name', city_name)}",
            "temp_c": round(float(data["main"]["temp"]), 1),
            "humidity": int(data["main"]["humidity"]),
            "description": data["weather"][0]["description"].title(),
            "icon": f"https://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png",
        }
        _weather_cache[key] = (now, result, None)
        return result, None

    except Exception as e:
        _weather_cache[key] = (now, None, f"Weather exception: {e}")
        return None, f"Weather exception: {e}"


def fetch_forecast(city_name: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    if not city_name:
        return None, None
    if not OPENWEATHER_API_KEY:
        return None, "OPENWEATHER_API_KEY is not set."

    key = _city_key(city_name)
    now = time.time()

    if key in _forecast_cache and now - _forecast_cache[key][0] < WEATHER_TTL_SEC:
        return _forecast_cache[key][1], _forecast_cache[key][2]

    try:
        params = {"q": city_name, "appid": OPENWEATHER_API_KEY, "units": "metric", "cnt": 8}
        r = requests.get(OWM_FORECAST, params=params, timeout=8)

        if r.status_code == 404 and "," not in city_name:
            params["q"] = f"{city_name},IN"
            r = requests.get(OWM_FORECAST, params=params, timeout=8)

        if r.status_code != 200:
            try:
                data = r.json()
                msg = data.get("message", f"HTTP {r.status_code}")
            except Exception:
                msg = f"HTTP {r.status_code}"
            _forecast_cache[key] = (now, None, f"Forecast error: {msg}")
            return None, f"Forecast error: {msg}"

        data = r.json()
        items = data.get("list", [])[:4]
        simplified = [{
            "time": it.get("dt_txt"),
            "temp_c": round(float(it["main"]["temp"]), 1),
            "humidity": int(it["main"]["humidity"]),
            "desc": it["weather"][0]["description"].title(),
            "icon": f"https://openweathermap.org/img/wn/{it['weather'][0]['icon']}@2x.png",
        } for it in items]

        _forecast_cache[key] = (now, simplified, None)
        return simplified, None

    except Exception as e:
        _forecast_cache[key] = (now, None, f"Forecast exception: {e}")
        return None, f"Forecast exception: {e}"


# --------------------------------
# Grad-CAM (safe build)
# --------------------------------
GRAD_MODEL = None


def build_gradcam_model():
    base = BASE_CNN
    if base is None or model is None:
        raise RuntimeError("Grad-CAM: model/base CNN not loaded.")

    last_conv_layer = base.get_layer("Conv_1")

    gap_layer = model.get_layer("global_average_pooling2d")
    drop_layer = model.get_layer("dropout")
    dense_layer = model.get_layer("dense")

    base_input = base.input
    x = base.output
    x = gap_layer(x)
    x = drop_layer(x, training=False)
    preds = dense_layer(x)

    grad_model = tf.keras.models.Model(inputs=base_input, outputs=[last_conv_layer.output, preds], name="gradcam_model")
    return grad_model


try:
    GRAD_MODEL = build_gradcam_model()
except Exception as e:
    print("⚠ Grad-CAM build skipped:", e)
    GRAD_MODEL = None


def generate_gradcam(image_path: str) -> str:
    if Image is None:
        raise RuntimeError("Pillow (PIL) missing. Install: pip install pillow")
    if GRAD_MODEL is None:
        raise RuntimeError("Grad-CAM not available (model missing/incompatible).")

    img = kimage.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = kimage.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0).astype("float32")

    with tf.GradientTape() as tape:
        conv_out, preds = GRAD_MODEL(img_arr)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1).numpy()

    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((IMG_SIZE, IMG_SIZE))
    heatmap_arr = np.array(heatmap_img)

    import matplotlib.cm as cm
    colored_heatmap = (cm.get_cmap("jet")(heatmap_arr / 255.0)[..., :3] * 255).astype(np.uint8)

    orig = np.array(img).astype("float32")
    alpha = 0.55
    overlay = (alpha * colored_heatmap + (1 - alpha) * orig).astype(np.uint8)

    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)
    out_path = os.path.join(UPLOAD_FOLDER, f"gradcam_{name}.png")
    Image.fromarray(overlay).save(out_path)
    return out_path.replace("\\", "/")


# --------------------------------
# NDVI Route (separate, safe)
# --------------------------------
@app.route("/ndvi/analyze", methods=["POST"])
def ndvi_analyze():
    if load_rgb_image is None or ndvi_metrics is None:
        return {"error": "NDVI module not available. Ensure ndvi_utils.py exists and install Pillow: pip install pillow"}, 500

    if "image" not in request.files:
        return {"error": "No image file provided (expected form field name: image)"}, 400

    file = request.files["image"]
    if file.filename == "":
        return {"error": "Empty filename"}, 400

    filename = secure_filename(file.filename)
    save_name = f"ndvi_{filename}"
    save_path = os.path.join("static", "uploads", save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)

    try:
        rgb = load_rgb_image(save_path)
        metrics = ndvi_metrics(rgb)
        return {"status": "ok", "image_path": save_path.replace("\\", "/"), **metrics}, 200
    except Exception as e:
        return {"error": f"NDVI analysis failed: {str(e)}"}, 500


# --------------------------------
# Soil API Route
# --------------------------------
@app.route("/soil/recommendations", methods=["POST"])
def soil_recommendations_api():
    data = request.get_json(silent=True) or {}
    state = (data.get("state") or "").strip()
    soil_type = (data.get("soil_type") or "").strip()

    chosen, info, source = choose_soil(state, soil_type)
    defaults = get_state_default_soils(state)

    return jsonify({
        "state": state,
        "chosen_soil": chosen,
        "source": source,
        "state_default_soils": defaults,
        "soil_info": info
    })


# ================================================
# OFFLINE LLM CHATBOT (Ollama) — /chat endpoint
# ================================================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3")  # you pulled phi3


def _chat_system_prompt() -> str:
    return (
        "You are 'Farming Help Assistant', an agriculture advisory chatbot inside a student PBL Flask app.\n"
        "Be concise, practical, and safe.\n"
        "Do NOT give exact pesticide dosages; advise to follow label + local agriculture officer guidance.\n"
        "Use provided context (disease label, severity, weather, soil traits, marketplace, laws).\n"
        "Output plain text with short bullet points.\n"
    )


def _format_context(ctx: dict) -> str:
    parts = []
    if ctx.get("result"): parts.append(f"Disease label: {ctx['result']}")
    if ctx.get("severity"): parts.append(f"Severity: {ctx['severity']}")
    if ctx.get("risk"): parts.append(f"Weather risk: {ctx['risk']}")

    w = ctx.get("weather")
    if isinstance(w, dict) and w:
        parts.append(f"Weather: {w.get('temp_c')}°C, humidity {w.get('humidity')}%, {w.get('description')}")
        parts.append(f"City: {w.get('city')}")

    if ctx.get("chosen_soil"):
        parts.append(f"Soil: {ctx['chosen_soil']}")

    si = ctx.get("soil_info")
    if isinstance(si, dict) and si.get("traits"):
        t = si["traits"]
        parts.append(f"Soil traits: water_retention={t.get('water_retention')}, drainage={t.get('drainage')}, ph_tendency={t.get('ph_tendency')}")

    if isinstance(ctx.get("laws_links"), list) and ctx["laws_links"]:
        parts.append(f"Laws links available: {len(ctx['laws_links'])} items")

    if isinstance(ctx.get("market"), list) and ctx["market"]:
        parts.append(f"Marketplace suggestions available: {len(ctx['market'])} items")

    return "\n".join(parts) if parts else "No analysis context available yet."


def _session_get_history():
    hist = session.get("chat_history", [])
    if not isinstance(hist, list):
        hist = []
    return hist[-10:]


def _session_append(role: str, content: str):
    hist = _session_get_history()
    hist.append({"role": role, "content": content})
    session["chat_history"] = hist[-10:]


def ollama_chat_reply(user_message: str, ctx: dict):
    user_message = (user_message or "").strip()
    if not user_message:
        return "Ask: 'What should I do now?' or 'Explain the risk'.", "fallback"

    history = _session_get_history()
    transcript_lines = []
    for m in history:
        r = (m.get("role") or "").upper()
        c = (m.get("content") or "").strip()
        if r and c:
            transcript_lines.append(f"{r}: {c}")
    transcript = "\n".join(transcript_lines)

    prompt = (
        f"SYSTEM:\n{_chat_system_prompt()}\n\n"
        f"CONTEXT:\n{_format_context(ctx)}\n\n"
        f"HISTORY:\n{transcript if transcript else '(none)'}\n\n"
        f"USER:\n{user_message}\n\n"
        f"ASSISTANT:\n"
    )

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 150
                }
            },
            timeout=60
        )
        if r.status_code != 200:
            return f"Ollama error HTTP {r.status_code}: {r.text}", "fallback"

        data = r.json()
        reply = (data.get("response") or "").strip()
        return (reply if reply else "No response from model."), "ollama"

    except Exception as e:
        return f"Ollama exception: {e}. Is Ollama running?", "fallback"


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "")

    ctx = {
        "result": data.get("result"),
        "severity": data.get("severity"),
        "risk": data.get("risk"),
        "weather": data.get("weather"),
        "chosen_soil": data.get("chosen_soil"),
        "soil_info": data.get("soil_info"),
        "market": data.get("market"),
        "laws_links": data.get("laws_links"),
    }

    _session_append("user", user_msg)
    reply, mode = ollama_chat_reply(user_msg, ctx)
    _session_append("assistant", reply)

    return jsonify({"reply": reply, "mode": mode})


@app.route("/chat/reset", methods=["POST"])
@login_required
def chat_reset():
    session["chat_history"] = []
    return jsonify({"status": "ok"})


# --------------------------------
# MAIN UI ROUTE (Protected)
# --------------------------------
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    result = None
    image_url = None
    image_path_internal = None
    gradcam_url = None
    recs: List[str] = []
    weather = None
    forecast = None
    weather_error = None
    forecast_error = None
    city = ""
    state = ""
    predicted_label = None
    market = []
    laws_links = []
    risk = None
    severity_ui = None
    soil_type = ""
    chosen_soil = None
    soil_info = None
    soil_source = None
    soil_defaults = []

    if request.method == "POST":
        action = request.form.get("action", "analyze")
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip() or ""

        if action == "analyze":
            soil_type = request.form.get("soil_type", "").strip()
            soil_defaults = get_state_default_soils(state)
            chosen_soil, soil_info, soil_source = choose_soil(state, soil_type)

            file = request.files.get("file")
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                image_path_internal = filepath
                image_url = filepath.replace("\\", "/")

                predicted_label = predict_label_multiclass(image_path_internal)
                result = predicted_label

                if city:
                    weather, weather_error = fetch_weather(city)
                    forecast, forecast_error = fetch_forecast(city)

                risk = compute_risk_level(
                    humidity=weather["humidity"] if weather else None,
                    temperature=weather["temp_c"] if weather else None
                )

                recs = get_advanced_recommendations(
                    predicted_label,
                    ndvi_delta=None,
                    humidity=weather["humidity"] if weather else None,
                    temperature=weather["temp_c"] if weather else None
                )
                recs += ["", "----", ""]
                recs += soil_weather_leaf_fusion_notes(soil_info, weather, predicted_label)

                market = marketplace_recommendations(predicted_label)
                laws_links = laws_for_state(state, predicted_label)

                if predicted_label:
                    label_lower = predicted_label.lower()
                    if "healthy" in label_lower:
                        severity_ui = "Normal"
                    else:
                        if risk and risk != "Unknown":
                            severity_ui = {"Low": "Mild", "Moderate": "Moderate", "High": "Severe", "Severe": "Severe"}.get(risk, "Moderate")
                        else:
                            severity_ui = determine_severity(None).capitalize()

        elif action == "gradcam":
            existing = request.form.get("image_path", "")
            city = request.form.get("city", "").strip()
            state = request.form.get("state", "").strip() or ""

            soil_type = request.form.get("soil_type", "").strip()
            soil_defaults = get_state_default_soils(state)
            chosen_soil, soil_info, soil_source = choose_soil(state, soil_type)

            if existing:
                existing = existing.replace("\\", "/")

            if existing and os.path.exists(existing):
                try:
                    image_path_internal = existing
                    image_url = existing.replace("\\", "/")

                    if city:
                        weather, weather_error = fetch_weather(city)
                        forecast, forecast_error = fetch_forecast(city)

                    risk = compute_risk_level(
                        humidity=weather["humidity"] if weather else None,
                        temperature=weather["temp_c"] if weather else None
                    )

                    predicted_label = predict_label_multiclass(image_path_internal)
                    result = predicted_label

                    recs = get_advanced_recommendations(
                        predicted_label,
                        ndvi_delta=None,
                        humidity=weather["humidity"] if weather else None,
                        temperature=weather["temp_c"] if weather else None
                    )
                    recs += ["", "----", ""]
                    recs += soil_weather_leaf_fusion_notes(soil_info, weather, predicted_label)

                    market = marketplace_recommendations(predicted_label)
                    laws_links = laws_for_state(state, predicted_label)

                    if predicted_label:
                        label_lower = predicted_label.lower()
                        if "healthy" in label_lower:
                            severity_ui = "Normal"
                        else:
                            if risk and risk != "Unknown":
                                severity_ui = {"Low": "Mild", "Moderate": "Moderate", "High": "Severe", "Severe": "Severe"}.get(risk, "Moderate")
                            else:
                                severity_ui = determine_severity(None).capitalize()

                    grad_path = generate_gradcam(image_path_internal)
                    gradcam_url = grad_path.replace("\\", "/")

                except Exception as e:
                    traceback.print_exc()
                    weather_error = f"Grad-CAM error: {e}"

    return render_template(
        "index.html",
        image=image_url,
        image_path_internal=image_path_internal,
        gradcam=gradcam_url,
        result=result,
        label=predicted_label,
        recommendations=recs,
        market=market,
        laws_links=laws_links,
        weather=weather,
        forecast=forecast,
        weather_error=weather_error,
        forecast_error=forecast_error,
        city=city or "",
        state=state or "",
        risk=risk,
        severity=severity_ui,
        soil_types=list(SOIL_CATALOG.keys()),
        soil_defaults=soil_defaults,
        soil_type=soil_type,
        chosen_soil=chosen_soil,
        soil_source=soil_source,
        soil_info=soil_info
    )


# --------------------------------
# Error handler
# --------------------------------
@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):
    return {"error": "File too large. Max 5MB."}, 413


# --------------------------------
# MAIN
# --------------------------------
if __name__ == "__main__":
    print("OPENWEATHER_API_KEY detected?", bool(os.environ.get("OPENWEATHER_API_KEY")))
    if not OPENWEATHER_API_KEY:
        print("NOTE: Weather disabled until API key is set.")

    print("Login user:", APP_USER)
    print("Ollama expected at:", OLLAMA_URL)
    print("Ollama model:", OLLAMA_MODEL)

    app.run(debug=True)
