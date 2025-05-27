# Nhận diện đeo khẩu trang bằng EfficientNetB0 và YOLOv8
## Thành viên thực hiện
Nghiêm Hồng Ngọc, Trần Ngọc Ánh, Đỗ Mai Phương, Trần Thanh Xuân
## Giới thiệu đề tài

Để phòng bị cho sự trở lại của đại dịch COVID-19 có thể bùng phát trong tương lai, nhóm chúng em thực hiện đề tài nhận diện việc đeo khẩu trang đúng cách. Đề tài áp dụng hai hướng tiếp cận học sâu phổ biến:

- **Image Classification (Phân loại ảnh)** với mô hình EfficientNetB0
- **Object Detection (Phát hiện đối tượng)** với mô hình YOLOv8

## Mục tiêu

- Phát hiện xem người trong ảnh/video có đeo khẩu trang hay không, hoặc đeo sai cách.
- So sánh hiệu quả của hai hướng tiếp cận classification và detection.
- Triển khai mô hình phát hiện thời gian thực, có khả năng ứng dụng trong các tình huống thực tế như cổng trường học, bệnh viện.

## Cấu trúc thư mục

MaskDetection/

├── data/ # Dữ liệu đầu vào (ảnh, annotation)

├── notebooks/ # Notebook huấn luyện mô hình

├── models/ # File model đã huấn luyện (.h5, .pt)

├── utils/ # Hàm hỗ trợ: vẽ ảnh, tính metrics...

├── demo/ # Mã nguồn demo mô hình real-time

├── report/ # Báo cáo, hình minh họa

├── requirements.txt # Thư viện cần cài đặt

├── README.md # Mô tả đề tài

└── .gitignore # Bỏ qua các file không cần thiết


## Dữ liệu sử dụng

- Dữ liệu **phân loại ảnh**: Lấy từ [Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- Dữ liệu **phát hiện đối tượng**: Lấy từ [Roboflow - msk2](https://universe.roboflow.com/mask-zmvem/msk2/dataset/1)

## Công nghệ sử dụng
Python 3.10

TensorFlow 2.x (EfficientNetB0)

Ultralytics YOLOv8

OpenCV, NumPy, Pandas, Matplotlib

TFLite, Roboflow, Google Colab

