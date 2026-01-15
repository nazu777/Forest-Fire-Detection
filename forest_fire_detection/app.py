import io
import base64
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()

# --- MODEL INITIALIZATION ---
def build_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

model = build_model()

# Load weights from WSL training
with open("model_weights.pkl", "rb") as f:
    weights_list = pickle.load(f)
    model.set_weights(weights_list)

# --- NEURAL WEIGHT EXPLORER ENGINE ---
def generate_weight_registry():
    """Generates a labeled, scrollable HTML registry of all model weights."""
    registry_html = ""
    # MobileNetV2 has many layers; we iterate through the saved weights list
    for i, w in enumerate(weights_list):
        layer_type = "Weight Matrix (W)" if len(w.shape) > 1 else "Bias Vector (b)"
        sample_values = str(w.flatten()[:10].tolist()) # Show first 10 values
        
        registry_html += f"""
        <div class="layer-card">
            <div class="layer-header">
                <span>LAYER_{i} | {layer_type}</span>
                <span class="shape-tag">{w.shape}</span>
            </div>
            <div class="stats-row">
                <span>MEAN: {np.mean(w):.6f}</span>
                <span>MAX: {np.max(w):.6f}</span>
                <span>MIN: {np.min(w):.6f}</span>
            </div>
            <div class="weight-scroll">
                <code>{sample_values}...</code>
            </div>
        </div>
        """
    return registry_html

# --- UI TEMPLATE ---
BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <style>
        :root {{ --primary: #00A699; --bg: #00171F; --card: #0a2530; }}
        body {{ margin: 0; background: var(--bg); color: white; font-family: 'Inter', sans-serif; display: flex; height: 100vh; overflow: hidden; }}
        
        /* Left Panel: Weight Registry */
        .registry-sidebar {{ width: 450px; background: rgba(0,0,0,0.3); border-right: 1px solid #333; padding: 20px; overflow-y: scroll; }}
        .layer-card {{ background: var(--card); border: 1px solid #1a3a4a; border-radius: 8px; padding: 12px; margin-bottom: 15px; }}
        .layer-header {{ display: flex; justify-content: space-between; font-weight: bold; color: var(--primary); font-size: 13px; margin-bottom: 8px; }}
        .shape-tag {{ background: #003459; padding: 2px 6px; border-radius: 4px; font-size: 10px; }}
        .stats-row {{ display: flex; justify-content: space-between; font-size: 11px; color: #888; margin-bottom: 8px; }}
        .weight-scroll {{ background: #000; padding: 8px; font-family: monospace; font-size: 10px; color: #00ff41; overflow-x: auto; white-space: nowrap; }}
        
        /* Right Panel: Main UI */
        .main-content {{ flex: 1; padding: 40px; overflow-y: auto; text-align: center; }}
        .search-box {{ background: #fff; padding: 15px; border-radius: 12px; display: flex; align-items: center; max-width: 600px; margin: 30px auto; }}
        input[type="file"] {{ color: #333; flex: 1; }}
        .btn {{ background: var(--primary); color: white; border: none; padding: 12px 25px; border-radius: 8px; font-weight: bold; cursor: pointer; }}
        .result-card {{ background: rgba(255,255,255,0.05); border: 1px solid #333; padding: 30px; border-radius: 16px; margin-top: 30px; }}
        h1 {{ font-size: 32px; letter-spacing: -1px; }}
        h1 span {{ color: var(--primary); }}
    </style>
</head>
<body>
    <div class="registry-sidebar">
        <h2 style="color: var(--primary);">NEURAL_REGISTRY</h2>
        <p style="font-size: 12px; color: #666;">Full Tensor Tracking for MobileNetV2</p>
        {weight_registry}
    </div>
    
    <div class="main-content">
        <h1>WILD<span>FIRE</span>.AI</h1>
        <p>Satellite Intelligence Dashboard v2.0</p>

        <form action="/predict/" enctype="multipart/form-data" method="post" class="search-box">
            <input name="file" type="file" required>
            <input type="submit" value="ANALYZE IMAGE" class="btn">
        </form>

        {results_section}
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return BASE_HTML.format(weight_registry=generate_weight_registry(), results_section="")

@app.post("/predict/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Inference
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    idx = np.argmax(prediction)
    
    label = "✅ SECURE: NO FIRE" if idx == 0 else "⚠️ WILDFIRE DETECTED"
    color = "#00A699" if idx == 0 else "#ff4b2b"
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    res_html = f"""
    <div class="result-card" style="border-top: 5px solid {color};">
        <h2 style="color: {color};">{label}</h2>
        <p>CONFIDENCE: {prediction[idx]*100:.2f}%</p>
        <img src="data:image/jpeg;base64,{img_str}" style="width: 100%; max-width: 400px; border-radius: 8px;">
        <div style="text-align: left; margin-top: 20px; background: #000; padding: 15px; font-family: monospace; font-size: 12px; color: {color};">
            <b>FINAL_PROBABILITY_TENSOR:</b><br>{str(prediction)}
        </div>
    </div>
    """
    
    return BASE_HTML.format(weight_registry=generate_weight_registry(), results_section=res_html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)