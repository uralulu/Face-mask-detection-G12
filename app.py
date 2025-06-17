import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from ultralytics import YOLO
import gradio as gr
from PIL import Image
import numpy as np
import cv2

# Load EfficientNetB0
model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 3)
model.load_state_dict(torch.load("best_efficientnet_b0.pth", map_location=torch.device("cpu")))
model.eval()

# Classes
class_names = ["mask_weared_incorrect", "with_mask", "without_mask"]

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load YOLOv8 model
yolo_model = YOLO("best_yolo_v8.pt")

# === Hàm kết hợp YOLOv8 và EfficientNetB0 ===
def detect_yolo_and_classify(image_pil):
    if image_pil is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    image_np = np.array(image_pil)
    annotated_img = image_np.copy()

    results = yolo_model(image_np)
    result = results[0]

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Sửa: Bo vùng crop để lấy đúng khuôn mặt hơn
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        x1_c = max(0, x1 + pad_w)
        y1_c = max(0, y1 + pad_h)
        x2_c = min(image_np.shape[1], x2 - pad_w)
        y2_c = min(image_np.shape[0], y2 - pad_h)

        face = image_np[y1_c:y2_c, x1_c:x2_c]
        if face.size == 0:
            continue

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            predicted_class_idx = np.argmax(probs)
            predicted_label = class_names[predicted_class_idx]

        color = (0, 255, 0)
        if predicted_label == "without_mask":
            color = (255, 0, 0)
        elif predicted_label == "mask_weared_incorrect":
            color = (0, 165, 255)

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, f"{predicted_label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return Image.fromarray(annotated_img)

# === Giao diện Gradio ===
custom_css = """
.output_image, .input_image {height: 400px !important; min-height: 200px !important;}
.gr-box {min-height: 400px !important;}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## Face Mask Detection Group 12 AI")
    gr.Markdown("Sản phẩm demo của nhóm 12 môn AI: YOLOv8 để phát hiện khuôn mặt và EfficientNetB0 để phân loại trạng thái khẩu trang")

    with gr.Tabs():
        with gr.TabItem("YOLOv8 + EfficientNetB0 Realtime"):
            gr.Markdown("### Kết hợp YOLOv8 (detect) và EfficientNetB0 (classify) với webcam realtime.")
            cam_input = gr.Image(type="pil", sources=["webcam"], streaming=True, label="Webcam Feed")
            cam_output = gr.Image(type="pil", label="Detection + Classification")
            gr.Interface(
                fn=detect_yolo_and_classify,
                inputs=cam_input,
                outputs=cam_output,
                live=True,
                allow_flagging="never"
            ).render()

# Khởi chạy
if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
