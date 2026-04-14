# Wan 2.2 Image-to-Video – RunPod Serverless

This repository contains a **Dockerized, serverless deployment** of the **Wan 2.2 Image-to-Video (I2V)** model, optimized for **RunPod Serverless GPU inference**.

⚠️ **Important assumption:**
This setup **expects the Wan 2.2 model files to already exist on disk** and be **mounted into the container** at:

```
/runpod-volume
```

This is the **recommended and production-safe setup** for large video generation models.

If the models are **not already present**, a helper script (`preload_model.py`) is provided to download **only the required files**.

This structure was ran and tested on **Nvidia A100 PCIe GPU**

---

## ✨ Features

* 🚀 Wan 2.2 Image-to-Video inference
* 🧠 Serverless GPU execution on RunPod
* 📦 Docker-based deployment
* 💾 Persistent model storage via RunPod volume
* 🎛 Request-level environment routing (staging / production)
* 🔐 Credential isolation per request
* 🧹 Automatic temp-file & GPU memory cleanup
* 🧩 Lightx2v LoRA support

---

## 🏗️ Architecture Overview

```text
Client Request
      ↓
RunPod Serverless Endpoint
      ↓
Docker Container (GPU)
      ↓
Mounted Model Volume (/runpod-volume/models)
      ↓
Wan 2.2 I2V Pipeline
      ↓
Video Generation
      ↓
Upload to S3
```

---

## 📁 Repository Structure

```text
.
├── app.py                  # RunPod serverless handler
├── preload_model.py        # Optional model downloader (NO model loading)
├── requirements.txt        # Python deps (excluding torch)
├── Dockerfile              # GPU-enabled Docker image
├── utils/
│   ├── utility.py          # Env routing + frame extraction
│   ├── s3.py               # S3 upload/download helpers
│   └── video.py            # Wan 2.2 pipeline loading & inference
├── .runpod/
│   └── hub.json             # RunPod Hub configuration
└── README.md
```

---

## 💾 Model Storage Strategy (Critical)

### ✅ Expected Setup (Recommended)

* Wan 2.2 model files are **already downloaded**
* A RunPod **persistent volume** is mounted at:

```
/runpod-volume
```

* The runtime loads models directly from this path
* No downloads occur during inference
* Startup is fast and stable

This is the **only reliable setup** for large Wan 2.2 models in production.

---

### ⚠️ Alternative: Download Models Using `preload_model.py`

If the models are **not already present**, you may use:

```bash
python preload_model.py
```

This script:

* Downloads **only the required Wan 2.2 files**
* Uses `hf_hub_download` (not `snapshot_download`)
* **Does NOT load models into memory**
* Avoids OOM during Docker build or runtime
* Stores files in the HuggingFace cache / configured model path

> ⚠️ Running this requires **~150GB available disk space**
> ⚠️ Not recommended for repeated or production cold starts

---

## 🧠 Runtime Model Loading

At runtime, models are loaded from disk:

```python
WanImageToVideoPipeline.from_pretrained(
    "/runpod-volume/models/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16
).to("cuda")
```

Because files are already present:

* No network downloads
* No startup delays
* Stable GPU memory usage

---

## ⚙️ Environment Configuration (RunPod)

All credentials are provided **via RunPod Hub**, not via `.env` files.

### Staging

```text
STAG_AWS_ACCESS_KEY_ID
STAG_AWS_SECRET_ACCESS_KEY
STAG_S3_BUCKET
```

### Production

```text
PROD_AWS_ACCESS_KEY_ID
PROD_AWS_SECRET_ACCESS_KEY
PROD_S3_BUCKET
```

### Shared

```text
AWS_REGION=us-east-2
```

Secrets are **never committed to GitHub**.

---

## 🎯 Request Payload Example

```json
{
  "input": {
    "level": "stag",
    "img_path": "s3://example-bucket/input/image.png",
    "clip_sec": 5,
    "prompts": [
      "A cinematic slow pan across a futuristic city at sunset",
      "The camera pulls back revealing flying vehicles and neon lights"
    ]
  }
}
```

---

## 🐳 Docker Notes

* CUDA base image: **12.8**
* Torch installed explicitly from PyTorch CUDA index
* Models **not loaded during build**
* Persistent volume expected at `/runpod-volume/models`

---

## 🚧 Known Constraints

* Wan 2.2 models are **very large**
* ~**150GB persistent storage** is required
* Docker builds must **never load models**
* GPU is only available at runtime, not build time

---

## 📌 Status

* ✅ Docker build stable
* ✅ Runtime inference working
* ✅ Volume-based model loading
* 🚧 Requires persistent storage provisioning on RunPod

---

## 📄 License

Provided **as-is** for experimental and personal use.
Model licenses remain with their respective authors.


 
