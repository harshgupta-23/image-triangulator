import cv2
import numpy as np
import gradio as gr
from sklearn.cluster import KMeans

def get_edge_mask(img_gray, n_levels=10):
    h, w = img_gray.shape
    pixels = img_gray.reshape(-1, 1).astype(np.float32)
    kmeans = KMeans(n_clusters=n_levels, n_init=10, random_state=42).fit(pixels)
    sorted_centers = np.sort(kmeans.cluster_centers_.flatten())
    segmented = sorted_centers[kmeans.labels_].reshape(h, w).astype(np.uint8)
    return cv2.Canny(segmented, 10, 100)

def get_vertices(mask, img_shape, percent=0.02):
    h, w = img_shape
    edge_pts = np.column_stack(np.where(mask > 0))
    if len(edge_pts) == 0: return np.array([[0,0], [w-1,0], [0,h-1], [w-1,h-1]], dtype=np.float32)
    n = max(int(len(edge_pts) * percent), 4)
    idx = np.random.default_rng(42).choice(len(edge_pts), min(n, len(edge_pts)), replace=False)
    pts = edge_pts[idx][:, ::-1].astype(np.float32)
    
    step = 1
    top    = np.column_stack([np.arange(0, w, step), np.zeros(len(np.arange(0, w, step)))])
    bottom = np.column_stack([np.arange(0, w, step), np.full(len(np.arange(0, w, step)), h - 1)])
    left   = np.column_stack([np.zeros(len(np.arange(0, h, step))), np.arange(0, h, step)])
    right  = np.column_stack([np.full(len(np.arange(0, h, step)), w - 1), np.arange(0, h, step)])
    return np.vstack([pts, top, bottom, left, right])

def render(img_rgb, points):
    h, w = img_rgb.shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    canvas = np.zeros_like(img_rgb)
    tri_mask = np.zeros((h, w), dtype=np.uint8)

    for t in subdiv.getTriangleList():
        pts = t.reshape(3, 2).astype(np.int32)
        pts[:, 0] = pts[:, 0].clip(0, w - 1)
        pts[:, 1] = pts[:, 1].clip(0, h - 1)
        tri_mask[:] = 0
        cv2.fillConvexPoly(tri_mask, pts, 255)
        avg_color = cv2.mean(img_rgb, mask=tri_mask)[:3]
        cv2.fillConvexPoly(canvas, pts, avg_color)
    return canvas

def low_poly_gradio(input_img, n_levels, percent):
    if input_img is None: return None
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    points = get_vertices(get_edge_mask(img_gray, int(n_levels)), img_gray.shape, percent/100)
    return render(input_img, points)

# --- Gradio Interface ---
demo = gr.Interface(
    fn=low_poly_gradio,
    inputs=[
        gr.Image(label="Upload Image"),
        gr.Slider(minimum=5, maximum=50, value=20, step=1, label="K-Means Levels"),
        gr.Slider(minimum=1, maximum=100.0, value=20, step=1, label="Vertex Density (%)")
    ],
    outputs=gr.Image(label="Low-Poly Result"),
    title="Low-Poly Art Generator",
    description="Upload an image to convert it into a triangulated low-poly masterpiece using K-Means and Delaunay Triangulation."
)

if __name__ == "__main__":
    demo.launch()