# Florence2 + SAM2 Two-Stage Inference Pipeline

🚀 This project integrates [Florence2](https://www.microsoft.com/en-us/research/project/florence/) and [SAM2 (Segment Anything Model v2)](https://segment-anything.com/) in a two-stage vision-language pipeline that performs advanced object understanding and segmentation.

---

## 🌐 Live Demo (Hugging Face)

Check out the live demo here:  
👉 [Hugging Face Space – Florence2 + SAM2](https://huggingface.co/spaces/SkalskiP/florence-sam)  
> Note: Hugging Face Spaces have **limited GPU time**, so for uninterrupted GPU access, use the Colab version or run locally.

---

## 🔗 Google Colab (GPU Required)

You can run this notebook on **Google Colab with GPU support**:  
👉 [Colab Deployment](https://github.com/DanialSoleimany/florence-sam-deploy/blob/main/florence_sam_deploy.ipynb)  

> ⚠️ **GPU is required** for this project to run properly. It will not work on CPU-only environments.

---

## 🔧 Features

- 🧠 Open-vocabulary object detection with Florence2
- 🖼️ Image captioning & phrase grounding
- ✂️ High-precision object segmentation with SAM2
- 🎯 Two-stage architecture: language-driven detection ➜ pixel-level segmentation

---

## 🧠 How It Works

### Stage 1: Florence2
- Task: Understands the image using text prompts
- Output: Bounding boxes, object names, or phrase locations

### Stage 2: SAM2
- Task: Takes Florence2 outputs and performs segmentation
- Output: Pixel-accurate masks for the detected regions

---

## 🖥️ Local Setup

### 🔃 Clone the repository

```bash
git clone https://github.com/DanialSoleimany/florence-sam-deploy.git
cd florence-sam-deploy
````

### 📦 Install dependencies

```bash
pip install -r requirements.txt
```

> 🛠️ Make sure your machine has a compatible **GPU** and the necessary CUDA drivers installed.

---

## 🧪 Example Use Cases

* 🔍 Open-world object recognition
* 🖌️ Smart image editing and cut-outs
* 🤖 Vision-language applications
* 🎨 AI-powered visual assistants

---

## 📁 Project Structure

```
├── florence_sam_deploy.ipynb     # Main notebook for inference
```

---

## 📷 Sample Results

| Prompt            | Output Mask (SAM2) |
| ----------------- | ------------------ |
| "A red car"       | ✅ Segmented mask   |
| "Dog on the left" | ✅ Segmented mask   |

> Add your own images and prompts to see dynamic segmentation results.

---

## 📚 References

* [Florence2 by Microsoft](https://www.microsoft.com/en-us/research/project/florence/)
* [SAM2: Segment Anything v2](https://segment-anything.com/)
* [Florence + SAM Hugging Face Demo](https://huggingface.co/spaces/SkalskiP/florence-sam)

---

## 📝 License

This project is intended for **research and educational purposes** only. Please check licenses of Florence2 and SAM2 for commercial use.


Would you like me to generate this as a downloadable `README.md` file for you too?
```
