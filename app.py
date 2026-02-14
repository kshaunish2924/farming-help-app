import os
import time
import json
import re
import traceback
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image as kimage

import os
os.makedirs("static/uploads", exist_ok=True)


# Optional: load .env for local dev
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except Exception:
    pass

# --------------------------------
# Flask setup & constants
# --------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Weather config
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
OWM_CURRENT = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"
WEATHER_TTL_SEC = 600  # 10 minutes cache

# Model config
MODEL_PATH = "plant_disease_model.keras"
IMG_SIZE = 224  # MobileNetV2 default

# --------------------------------
# Load model once
# --------------------------------
model = keras.models.load_model(MODEL_PATH)

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
        BASE_CNN = model  # fallback

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
    """Predicts exact class name from multiclass model."""
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
    """
    Simple weather-based risk scoring system.
    Returns: 'Low', 'Moderate', 'High', 'Severe', or 'Unknown'
    """
    if humidity is None:
        return "Unknown"

    # Base score from humidity
    if humidity < 60:
        score = 1
    elif humidity < 75:
        score = 2
    elif humidity < 85:
        score = 3
    else:
        score = 4

    # Temperature boost (fungal disease risk)
    if temperature is not None and 22 <= temperature <= 30:
        score = min(score + 1, 4)

    levels = {
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Severe"
    }
    return levels.get(score, "Unknown")

# --------------------------------
# Severity mapping (for NDVI etc.)
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

    # Normalize label key
    key = label.lower()
    key = key.replace(" ", "_")
    key = re.sub(r"_+", "_", key)  # collapse multiple underscores into one

    # ---------------- HEALTHY CASE ----------------
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

    # ---------------- DISEASE CASE ----------------
    matched = None
    for disease in DISEASE_RULES.keys():
        if disease in key:
            matched = disease
            break

    if not matched:
        return ["Disease detected, but no detailed rules available."]

    rules = DISEASE_RULES[matched]
    severity = determine_severity(ndvi_delta)  # currently NDVI-based; can later plug real NDVI

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
# Weather Functions (with cache)
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
# Grad-CAM (MobileNetV2-aware)
# --------------------------------
# --------------------------------
# Grad-CAM (Fixed for MobileNetV2)
# --------------------------------
# --------------------------------
# Grad-CAM (fixed for your MobileNetV2 model)
# --------------------------------
import tensorflow as tf

def build_gradcam_model():
    """
    Build a clean model for Grad-CAM:
    input -> MobileNetV2 -> GAP -> Dropout -> Dense
    while tapping the last Conv layer ('Conv_1') from MobileNetV2.
    """
    base = BASE_CNN
    if base is None:
        raise RuntimeError("Grad-CAM: BASE_CNN (mobilenetv2_1.00_224) not found.")

    # Last conv inside MobileNetV2
    try:
        last_conv_layer = base.get_layer("Conv_1")
    except Exception as e:
        raise RuntimeError(f"Grad-CAM: could not find Conv_1 layer: {e}")

    # Head layers from the full model
    gap_layer = model.get_layer("global_average_pooling2d")
    drop_layer = model.get_layer("dropout")
    dense_layer = model.get_layer("dense")

    # Build the functional graph starting from base CNN input
    base_input = base.input             # (None, 224, 224, 3)
    x = base.output                     # feature maps (7x7x1280)
    x = gap_layer(x)
    x = drop_layer(x, training=False)   # ensure inference mode
    preds = dense_layer(x)              # logits / softmax

    grad_model = tf.keras.models.Model(
        inputs=base_input,
        outputs=[last_conv_layer.output, preds],
        name="gradcam_model"
    )
    return grad_model, last_conv_layer

GRAD_MODEL, LAST_CONV_LAYER = build_gradcam_model()

def generate_gradcam(image_path: str) -> str:
    """
    Generate a colored Grad-CAM heatmap (JET colormap) for the given image
    and save it under static/uploads/gradcam_<originalname>.png
    """
    # 1) Load and preprocess image same as prediction
    img = kimage.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = kimage.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0).astype("float32")

    # 2) Forward + gradients through GRAD_MODEL
    with tf.GradientTape() as tape:
        conv_out, preds = GRAD_MODEL(img_arr)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)[0]              # (H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))     # (C,)

    conv_out = conv_out[0]                                # (H, W, C)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1).numpy()

    # 3) Normalize heatmap 0..1
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    # 4) Resize heatmap to image size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize((IMG_SIZE, IMG_SIZE))
    heatmap_arr = np.array(heatmap_img)

    # 5) Apply JET colormap (blue→green→yellow→red)
    import matplotlib.cm as cm
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap_arr / 255.0)       # RGBA
    colored_heatmap = (colored_heatmap[..., :3] * 255).astype(np.uint8)

    # 6) Overlay on original image
    orig = np.array(img).astype("float32")
    alpha = 0.55   # increase for stronger heatmap
    overlay = (alpha * colored_heatmap + (1 - alpha) * orig).astype(np.uint8)

    # 7) Save output
    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)
    out_path = os.path.join(UPLOAD_FOLDER, f"gradcam_{name}.png")
    Image.fromarray(overlay).save(out_path)

    return out_path.replace("\\", "/")


# --------------------------------
# ROUTES
# --------------------------------
@app.route("/", methods=["GET", "POST"])
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
    severity_ui = None  # for UI display (weather-based / fallback)

    if request.method == "POST":
        action = request.form.get("action", "analyze")
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip() or ""

        # ------------------- ANALYZE -------------------
        if action == "analyze":
            file = request.files.get("file")
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # normalize paths
                image_path_internal = filepath
                image_url = filepath.replace("\\", "/")

                predicted_label = predict_label_multiclass(image_path_internal)
                result = predicted_label

                # Weather first (for risk & recs)
                if city:
                    weather, weather_error = fetch_weather(city)
                    forecast, forecast_error = fetch_forecast(city)

                # Risk based on weather
                risk = compute_risk_level(
                    humidity=weather["humidity"] if weather else None,
                    temperature=weather["temp_c"] if weather else None
                )

                # Recommendations (environment-aware)
                recs = get_advanced_recommendations(
                    predicted_label,
                    ndvi_delta=None,
                    humidity=weather["humidity"] if weather else None,
                    temperature=weather["temp_c"] if weather else None
                )

                # Marketplace & laws
                market = marketplace_recommendations(predicted_label)
                laws_links = laws_for_state(state, predicted_label)

                # Severity for UI (D: show everywhere)
                if predicted_label:
                    label_lower = predicted_label.lower()
                    if "healthy" in label_lower:
                        severity_ui = "Normal"
                    else:
                        if risk and risk != "Unknown":
                            map_risk_to_severity = {
                                "Low": "Mild",
                                "Moderate": "Moderate",
                                "High": "Severe",
                                "Severe": "Severe",
                            }
                            severity_ui = map_risk_to_severity.get(risk, "Moderate")
                        else:
                            severity_ui = determine_severity(None).capitalize()

        # ------------------- GRAD-CAM -------------------
        elif action == "gradcam":
            existing = request.form.get("image_path", "")
            city = request.form.get("city", "").strip()
            state = request.form.get("state", "").strip() or ""

            if existing:
                existing = existing.replace("\\", "/")

            if existing and os.path.exists(existing):
                try:
                    image_path_internal = existing
                    image_url = existing.replace("\\", "/")

                    # Weather (if city provided)
                    if city:
                        weather, weather_error = fetch_weather(city)
                        forecast, forecast_error = fetch_forecast(city)

                    # Risk
                    risk = compute_risk_level(
                        humidity=weather["humidity"] if weather else None,
                        temperature=weather["temp_c"] if weather else None
                    )

                    # Predictions & recs
                    predicted_label = predict_label_multiclass(image_path_internal)
                    result = predicted_label

                    recs = get_advanced_recommendations(
                        predicted_label,
                        ndvi_delta=None,
                        humidity=weather["humidity"] if weather else None,
                        temperature=weather["temp_c"] if weather else None
                    )

                    market = marketplace_recommendations(predicted_label)
                    laws_links = laws_for_state(state, predicted_label)

                    # Severity for UI
                    if predicted_label:
                        label_lower = predicted_label.lower()
                        if "healthy" in label_lower:
                            severity_ui = "Normal"
                        else:
                            if risk and risk != "Unknown":
                                map_risk_to_severity = {
                                    "Low": "Mild",
                                    "Moderate": "Moderate",
                                    "High": "Severe",
                                    "Severe": "Severe",
                                }
                                severity_ui = map_risk_to_severity.get(risk, "Moderate")
                            else:
                                severity_ui = determine_severity(None).capitalize()

                    # Generate Grad-CAM
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
        severity=severity_ui
    )

# --------------------------------
# MAIN
# --------------------------------
if __name__ == "__main__":
    print("API KEY DETECTED?", bool(os.environ.get("OPENWEATHER_API_KEY")))
    if not OPENWEATHER_API_KEY:
        print("NOTE: Weather disabled until API key is set.")
    app.run(debug=True)
