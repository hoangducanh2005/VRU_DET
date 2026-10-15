Hãy triển khai **CSP (Center and Scale Prediction)** để chạy inference phát hiện pedestrian trên dataset ảnh đã lưu trong máy.

**Repository:** https://github.com/wang-xinyu/csp.pytorch

**Yêu cầu:**
1. Clone repo và thiết lập môi trường Python/PyTorch tương thích với CUDA/GPU hiện tại.
2. Tải pretrained checkpoint CSP ResNet-50 (CityPersons, Height + Offset): `net_e121_l0.hdf5`.
3. Load model và pretrained weights, chạy thử inference trên một ảnh.
4. Viết script `inference_folder.py` để inference toàn bộ ảnh trong thư mục dataset local (đường dẫn cấu hình được).
5. Thực hiện preprocessing đúng theo repo, decode bounding boxes và vẽ bounding box kèm confidence score.
6. Lưu ảnh kết quả vào thư mục `output/`, đồng thời in số pedestrian phát hiện trên mỗi ảnh.
7. Viết README hướng dẫn cài đặt và chạy inference.

**Lưu ý:**
- Chỉ inference, không training/fine-tuning.
- Ưu tiên sử dụng code và hàm có sẵn trong repository.
- Không sửa kiến trúc model.
- Kiểm tra tính tương thích của checkpoint, dependency và kích thước ảnh; không tự giả định API.
- Báo rõ các bước không thể thực hiện, đặc biệt nếu không tải được checkpoint.

**Dataset:** Shard 5 navsim mini ở local : D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311

**Mục tiêu:** Chạy thành công CSP pretrained trên dataset riêng để đánh giá pedestrian detection phục vụ nghiên cứu VRU detection trong autonomous driving.

**Side**  Yêu cầu 2,3 hướng dẫn tôi tự làm. 
**Đã tạm dừng ở bước tải weight ( chưa tải )**

------------------------------------------------------------
Với những gì đã tải , làm task 2 
Triển khai **VRU Detection & Tracking** sử dụng model [YOLO26n-BDD100K](https://github.com/lochuan/yolo26n-bdd100k).

**Yêu cầu:**
1. Clone repo, cài đặt dependencies và tải pretrained weights (ưu tiên FP32 để chạy thử).
2. Chạy inference trên video/dataset có sẵn trong máy, không training.
3. Detect các class VRU: `person`, `rider`, `bicycle`, `motorcycle`.
4. Tích hợp **ByteTrack** để gán và duy trì `track_id` giữa các frame.
5. Visualize bounding box, class, confidence và track ID trên video.
6. Lưu video kết quả và file CSV chứa `[frame_id, track_id, class, confidence, bbox]`.
7. Đo **FPS, inference latency (ms/frame), end-to-end latency**.
8. Code Python/PyTorch rõ ràng, dễ chuyển đổi sang BoT-SORT hoặc ONNX/TensorRT sau này.

**Ưu tiên:** GPU nếu có, latency thấp, tracking ổn định, không chỉnh sửa kiến trúc model gốc. Kiểm tra cấu trúc repo và checkpoint trước khi triển khai.
**Dataset:** Shard 5 navsim mini ở local : D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311
**Side**: hướng dẫn tôi tự làm yêu cầu 1.


------------------------------------------------------------
Task 3 : 
Triển khai **VRU Detection & Tracking** bằng YOLO26m cho bài toán Autonomous Driving.

**Yêu cầu:**
0. Gom code, kết quả của từng model vào các thư mục khác nhau ( YOLO26n và metrics của nó riêng, 26m riêng)
1. Sử dụng checkpoint pretrained **YOLO26m-KITTI**: https://huggingface.co/dronefreak/kitti-yolo26m (ưu tiên) hoặc YOLO26m-COCO.
2. Tải weights, cài dependencies và cấu hình GPU/CUDA.
3. Chạy inference trên video/dataset có sẵn trong máy, **không training/fine-tuning**.
4. Detect các class VRU: `Pedestrian`, `Cyclist` (tùy checkpoint).
5. Tích hợp **ByteTrack** để duy trì `track_id` qua các frame.
6. Visualize bounding box, class, confidence, track ID;
7. Lưu CSV: `[frame_id, track_id, class, confidence, x1, y1, x2, y2]`.
8. Đo FPS, inference latency và end-to-end latency (ms/frame).
9. So sánh YOLO26m với YOLO26n đã triển khai trên cùng video: chất lượng detection, số trường hợp bỏ sót VRU, FPS và latency. ( để sau chưa cần làm)

**Deliverables**
- Chạy inference , lưu 1 folder chứa các frame ảnh đã visualize, không cần ghép thành video. 
- Các frame phải liên tiếp nhau, tránh khác hẳn timestamp như đợt chạy task2. 
- Đưa ra evidence score hệt như task2_vru_detection.md 


**Lưu ý:** Giữ nguyên cấu trúc project hiện tại, tái sử dụng pipeline tracking, chỉ thay model/checkpoint. Không tự động thay đổi dataset hoặc code không liên quan. Ưu tiên độ chính xác VRU và tracking ổn định, đồng thời đảm bảo latency thấp.
--------------------------------------------------------------------------------------------

Task 4 
Implement and run a **VRU Detection baseline using YOLO11s** for an 8-camera autonomous driving system.

**Requirements:**
1. **Model:** Ultralytics YOLO11s with COCO pretrained weights (`yolo11s.pt`). No fine-tuning.
2. **Input:** 8 independent camera images, processed using one shared YOLO11s model (batch inference). Dataset to run inference : navsim . run all 8 camera, ví visualize deliverables : 1 frame = goc nhin tu 8 camera tai cung 1 timestamp co hien thi boundingbox. Dong thoi tao metric danh gia luot chay inference nay
3. **Detection:** Focus on VRU classes: `person`, `bicycle`, `motorcycle`.
4. **Configuration:** `imgsz=640`, GPU NVIDIA CUDA
5. **Benchmark:** Measure throughput (images/sec), per-batch latency, p95 latency, GPU memory usage, and FPS per camera equivalent.
6. **Evaluation:** Use sample driving images or an available labeled dataset to measure Precision, Recall, mAP50, and mAP50-95. Clearly distinguish labeled evaluation from qualitative inference.
7. **Visualization:** Save detection results for all 8 cameras with bounding boxes, confidence scores, and camera IDs.
8. **Deliverables:** Runnable Python scripts,evidence .md, benchmark results (CSV/JSON), and visualized outputs.

**Constraints:** Do not fine-tune or modify YOLO11 architecture. Do not require downloading the full BDD100K dataset. Use real measurements only; never fabricate metrics. Ensure the baseline is reproducible.