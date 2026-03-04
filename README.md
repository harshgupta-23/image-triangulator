---
title: Image Triangulator
emoji: 🦀
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 6.8.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Transform images into geometric low-poly art using K-Means.
---


## 🎨 Low-Poly Art Generator

This interactive tool transforms standard images into stylized, geometric "low-poly" art. It uses a combination of machine learning and computational geometry to create unique triangulated meshes.

## 🖼️ Examples

| Original Image | Low-Poly Transformation |
| :---: | :---: |
| ![Roses Original](roses_original.jpg) | ![Roses Transformed](roses_triangulated.webp) |
| ![Lake Original](lake_original.jpg) | ![Lake Transformed](lake_triangulated.webp) |

---

## 🚀 How it Works

The application processes images through a four-stage pipeline:

1.  **Color Quantization:** Using **K-Means Clustering**, the image's colors are simplified into a user-defined number of levels (*K*). This creates the "posterized" look that serves as the base for the art.
2.  **Edge Detection:** A **Canny** edge detector identifies the structural contours of the subject.
3.  **Vertex Sampling:** Points are sampled from the detected edges. To ensure the entire image is covered, dense vertices are also placed along the four borders and corners of the frame.
4.  **Delaunay Triangulation:** Using `cv2.Subdiv2D`, the sampled points are connected into a non-overlapping mesh of triangles. Each triangle is then filled with the average color of the original pixels it covers.



---

## 🛠️ Installation & Local Usage

To run this project on your own machine:

1. **Clone the repository:**


   You can find the source here: [image-triangulator](https://huggingface.co/spaces/harshg23/image-triangulator)


   ```bash
   git clone https://huggingface.co/spaces/harshg23/image-triangulator
   cd image-triangulator

2. **Install dependencies:**
   ```bash
   pip install gradio opencv-python-headless scikit-learn numpy

3. **Launch the app:**
   ```bash
   python app.py
