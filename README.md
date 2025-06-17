# Nhận diện đeo khẩu trang bằng EfficientNetB0 và YOLOv8
## Thành viên thực hiện
Nghiêm Hồng Ngọc, Trần Ngọc Ánh, Đỗ Mai Phương, Trần Thanh Xuân
## Giới thiệu đề tài

Để phòng bị cho sự trở lại của đại dịch COVID-19 có thể bùng phát trong tương lai, nhóm chúng em thực hiện đề tài nhận diện việc đeo khẩu trang đúng cách. Đề tài áp dụng hai hướng tiếp cận học sâu phổ biến:

- **Image Classification (Phân loại ảnh)** với mô hình EfficientNetB0
- **Object Detection (Phát hiện đối tượng)** với mô hình YOLOv8

## Mục tiêu

- Phát hiện xem người trong ảnh/video có đeo khẩu trang hay không, hoặc đeo sai cách.
- So sánh hiệu quả của hai hướng tiếp cận classification và detection. => đê ra hướng kết hợp: YOLOv8 xác định mặt, EfficientNetB0 nhận diện trạng thái
- Triển khai mô hình trên webcam realtime, có khả năng ứng dụng trong các tình huống thực tế như cổng trường học, bệnh viện.

## Dữ liệu sử dụng

- Bộ dataset huấn luyện: Lấy từ [Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

## Công nghệ sử dụng
Python

TensorFlow

Ultralytics YOLOv8

OpenCV, NumPy, Pandas, Matplotlib

HuggingFace, Roboflow, Kaggle, Google Colab

## Sản phẩm Demo

2 Mô hình riêng biệt: https://huggingface.co/spaces/uralulu/face-mask-detector
Kết hợp: https://huggingface.co/spaces/uralulu/face-mask-detector-upgrade
