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