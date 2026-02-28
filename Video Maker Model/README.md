# 🎬 Video Maker Model — AI Video Generator

An AI-powered **text-to-video generation pipeline** using Stable Diffusion, with built-in **prompt guardrails** and a Gradio web interface.

---

## ✨ Features

- **Text-to-Video Pipeline:**
  1. **Prompt Guard** — Validates user prompt using keyword matching + semantic similarity
  2. **Prompt Refiner** — Enhances the prompt for better generation
  3. **Frame Generator** — Generates image frames using Stable Diffusion
  4. **Post-Processing** — Stitches frames into a video using FFmpeg
- **Safety Guardrails** — Only allows prompts related to vintage/classic cars (customizable)
- **Gradio UI** — Clean web interface for prompt input and video playback

---

## 🏗️ Project Structure

```
Video Maker Model/
├── app.py               # Main pipeline orchestrator
├── prompt_guard.py       # Semantic + keyword-based prompt validation
├── prompt_refiner.py     # Prompt enhancement
├── video_generator.py    # Stable Diffusion frame generation
├── postprocess.py        # FFmpeg video stitching
├── ui.py                 # Gradio web interface
└── requirements.txt      # Dependencies
```

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the App
```bash
python app.py
```

The Gradio interface will open in your browser.

---

## ⚙️ How It Works

```
User Prompt → Prompt Guard (safety check) → Prompt Refiner → Stable Diffusion (frame gen) → FFmpeg (video) → Output
```

### Prompt Guard
Uses a dual-gate system:
1. **Keyword Gate** — Fast check against allowed keywords
2. **Semantic Gate** — Cosine similarity using `all-MiniLM-L6-v2` sentence embeddings

---

## 🧠 Tech Stack
- **Model:** Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Video:** OpenCV + FFmpeg
- **UI:** Gradio
- **Framework:** PyTorch, Diffusers, Transformers
