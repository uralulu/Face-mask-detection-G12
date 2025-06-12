import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import gradio as gr
from PIL import Image
import numpy as np
import mediapipe as mp
import cv2 # Cần thư viện này để vẽ lên ảnh

# Load model
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

# Mediapipe face detection (khởi tạo một lần và tái sử dụng)
face_detection_model = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --- Hàm dự đoán chính (sẽ dùng cho cả webcam và upload) ---
# Hàm này cần phải hoạt động linh hoạt cho cả live stream và ảnh tĩnh
def predict(image_pil): # Hàm predict sẽ nhận một PIL Image
    if image_pil is None:
        print("⚠️ Image is None (from Gradio)")
        return np.zeros((480, 640, 3), dtype=np.uint8), {name: 0.0 for name in class_names}

    image_np = np.array(image_pil) # Chuyển PIL Image sang NumPy array (RGB)
    annotated_image = image_np.copy() # Tạo bản sao để vẽ

    try: # Thêm khối try-except để bắt lỗi
        # Sử dụng đối tượng face_detection_model đã khởi tạo global
        results = face_detection_model.process(image_np)

        # Nếu không tìm thấy khuôn mặt
        if not results.detections:
            cv2.putText(annotated_image, "No Face Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return annotated_image, {"No face detected": 1.0}

        # Lặp qua các khuôn mặt được phát hiện (ở đây chỉ xử lý cái đầu tiên để đơn giản)
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image_np.shape # Chiều cao và chiều rộng của ảnh gốc
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                     int(bboxC.width * iw), int(bboxC.height * ih)

        # Đảm bảo tọa độ bounding box nằm trong giới hạn ảnh
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)

        # Cắt vùng khuôn mặt từ ảnh PIL ban đầu (tránh các vấn đề về định dạng màu)
        face_image_cropped_pil = image_pil.crop((x1, y1, x2, y2))

        # Kiểm tra nếu ảnh cắt ra bị rỗng (ví dụ, bounding box quá nhỏ)
        if face_image_cropped_pil.size[0] == 0 or face_image_cropped_pil.size[1] == 0:
            cv2.putText(annotated_image, "Invalid Face Crop", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return annotated_image, {"Invalid face crop": 1.0}

        input_tensor = transform(face_image_cropped_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]

        # Lấy nhãn dự đoán và xác suất
        predicted_class_idx = np.argmax(probs)
        predicted_label = class_names[predicted_class_idx]
        confidence = probs[predicted_class_idx]

        # Chọn màu cho khung và text
        color = (0, 255, 0) # Green (RGB) for "with_mask"
        if predicted_label == "without_mask":
            color = (255, 0, 0) # Red (RGB) for "without_mask"
        elif predicted_label == "mask_weared_incorrect":
            color = (0, 165, 255) # Orange (RGB) for "mask_weared_incorrect"

        # Vẽ bounding box và text lên ảnh
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, f"{predicted_label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        return annotated_image, {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    except Exception as e:
        print(f"🔴 Error in predict: {e}")
        cv2.putText(annotated_image, f"Error: {str(e)[:50]}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return annotated_image, {"Processing Error": 1.0}

# --- Định nghĩa Gr.Interface cho Webcam (GIỮ NGUYÊN) ---
webcam_interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", sources=["webcam"], streaming=True, label="Webcam Feed"),
    outputs=[
        gr.Image(type="numpy", label="Detected Face with Prediction"),
        gr.Label(num_top_classes=3, label="Prediction Probability")
    ],
    live=True,
    title="Webcam Live Stream", # Title riêng cho interface này
    description="Live face mask detection from webcam.",
    allow_flagging="never"
)

# --- Bắt đầu cấu trúc Gr.Blocks để chứa Tabs ---
custom_css = """
.output_image, .input_image {height: 400px !important; min-height: 200px !important;}
.gr-box {min-height: 400px !important;}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## Face Mask Detection (EfficientNetB0 + MediaPipe)")
    gr.Markdown("Detects face mask status using PyTorch and EfficientNetB0. Choose a tab below for **Webcam Live Stream** or **Upload/Paste Image**.")
    gr.Markdown("Sản phẩm demo của nhóm 12 môn AI, sử dụng mô hình EfficientNet (Image Classification).")
    with gr.Tabs():
        # --- TAB 1: Webcam Live Stream (Nhúng gr.Interface đã định nghĩa) ---
        with gr.TabItem("Webcam Live Stream"):
            webcam_interface.render() # Đây là cách để nhúng gr.Interface vào gr.Blocks

        # --- TAB 2: Upload/Paste Image ---
        with gr.TabItem("Upload/Paste Image"):
            gr.Markdown("### Upload an image or paste from clipboard for prediction.")
            with gr.Row():
                upload_image_input = gr.Image(type="pil", sources=["upload", "clipboard"], label="Upload or Paste Image")
                upload_output_image = gr.Image(type="numpy", label="Detected Face with Prediction")
            upload_output_label = gr.Label(num_top_classes=3, label="Prediction Probability")
            
            # Nút Submit cho tab này
            upload_button = gr.Button("Submit Image")

            # Khi nút submit được nhấn, gọi hàm predict cho ảnh đã upload/paste
            upload_button.click(
                fn=predict, # Hàm predict sẽ xử lý ảnh tĩnh
                inputs=upload_image_input,
                outputs=[upload_output_image, upload_output_label],
                api_name="upload_predict" # Tùy chọn: đặt tên API
            )
            
            # Tùy chọn: Kích hoạt dự đoán khi ảnh được tải lên (không cần nút Submit)
            # upload_image_input.change(
            #     fn=predict,
            #     inputs=upload_image_input,
            #     outputs=[upload_output_image, upload_output_label]
            # )

# Chạy ứng dụng Gradio
demo.launch(share=True, show_error=True)