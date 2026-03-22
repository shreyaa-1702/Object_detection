# Object Detection Using TensorFlow Hub

A deep learning project that performs real-time object detection on images using pre-trained models from **TensorFlow Hub**. This project demonstrates how to load, run, and visualize results from state-of-the-art object detection architectures including **Faster R-CNN** and **SSD MobileNet V2**.

---

## 📖 Project Overview

This project uses pre-trained object detection models available on TensorFlow Hub to identify and localize objects within images. The pipeline downloads an input image, runs it through a selected detection model, and overlays labeled bounding boxes with confidence scores on the output image.

Two model options are supported — one optimized for **accuracy** and one for **speed** — making it suitable for both research and lightweight deployment scenarios.

---

## 📂 Repository Structure

```
object-detection/
│
├── object_detection.ipynb       # Main Jupyter Notebook with full implementation
├── README.md                    # Project overview and instructions
├── requirements.txt             # Python dependencies
└── .gitignore                   # Files excluded from version control
```

---

## 🔍 Models Used

Two pre-trained object detection models from TensorFlow Hub are supported:

| Model | TFHub Module | Best For |
|-------|-------------|----------|
| **Faster R-CNN + InceptionResNet V2** | `faster_rcnn/openimages_v4/inception_resnet_v2/1` | High accuracy detection |
| **SSD + MobileNet V2** | `openimages_v4/ssd/mobilenet_v2/1` | Fast, lightweight inference |

> Both models are trained on the **Open Images V4** dataset, capable of detecting hundreds of object categories.

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|----------|-------------------|
| **Language** | Python 3.x |
| **Deep Learning** | TensorFlow, TensorFlow Hub |
| **Image Processing** | Pillow (PIL), NumPy |
| **Visualization** | Matplotlib |
| **Utilities** | `six`, `tempfile`, `time` |

---

## 🚀 Implementation Steps

### 1. Environment Setup
- Import TensorFlow, TensorFlow Hub, PIL, and visualization libraries
- Verify TensorFlow version and GPU availability

### 2. Image Downloading & Preprocessing
- Download image from a URL using `urllib`
- Resize the image to target dimensions using `ImageOps.fit()`
- Save as JPEG for model input

### 3. Model Loading
- Load the selected pre-trained detector from TensorFlow Hub
- Access the `default` signature for inference

### 4. Object Detection Inference
- Convert image to `float32` tensor
- Run the detector and record inference time
- Extract detection boxes, class names, and confidence scores

### 5. Bounding Box Visualization
- Draw color-coded bounding boxes around detected objects
- Overlay class labels with confidence scores (e.g., `Person: 94%`)
- Display the annotated image using Matplotlib

---

## 🔧 Key Functions

| Function | Description |
|----------|-------------|
| `download_and_resize_image()` | Downloads image from URL and resizes to specified dimensions |
| `display_image()` | Renders image using Matplotlib |
| `load_img()` | Reads and decodes a JPEG image using TensorFlow I/O |
| `draw_bounding_box_on_image()` | Draws a single labeled bounding box on a PIL image |
| `draw_boxes()` | Overlays all detected bounding boxes above a minimum confidence threshold |
| `run_detector()` | End-to-end pipeline — loads image, runs inference, and displays results |

---

## 📊 Output

- **Number of objects detected** printed to console
- **Inference time** displayed for performance benchmarking
- **Annotated image** rendered with:
  - Color-coded bounding boxes per object class
  - Class label and confidence score on each box

---

## 🏁 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/shreyaa-1702/object-detection.git
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook object_detection.ipynb
   ```

4. **Run all cells in order**
   - Optionally switch between Faster R-CNN and SSD MobileNet by changing `module_handle`

> 💡 **GPU Recommended** — The notebook is configured to run with GPU acceleration. On Google Colab, set Runtime → Change Runtime Type → **GPU** for faster inference.

---

## ⚙️ Requirements

```
tensorflow
tensorflow-hub
tensorflow-datasets
Pillow
numpy
matplotlib
six
```

Install all at once:
```bash
pip install tensorflow tensorflow-hub Pillow numpy matplotlib six
```

---

## 💡 Key Insights

- **Faster R-CNN + InceptionResNet V2** delivers high accuracy but slower inference — best for offline batch processing
- **SSD + MobileNet V2** runs significantly faster — suitable for near real-time or edge device applications
- Confidence threshold is set to `0.1` by default — adjustable in `draw_boxes()` via `min_score` parameter
- Maximum boxes displayed per image is `10` — configurable via `max_boxes` parameter

---

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## 🙋 Contributing

Contributions and suggestions are welcome. Feel free to open an issue or submit a pull request for any improvements or enhancements.
