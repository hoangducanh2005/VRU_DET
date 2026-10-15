# Kế hoạch preprocessing ảnh NavSim cho YOLO26m nuImages

## Mục tiêu và phạm vi

Kế hoạch này chuẩn bị ảnh camera NavSim trước inference bằng checkpoint YOLO26m fine-tune trên nuImages, class `pedestrian`. Ảnh NavSim có thể khác nuImages về camera, kích thước, độ sáng và tỷ lệ người đi bộ trong khung hình. Vì vậy preprocessing cần giữ nguyên nội dung hình học, phát hiện lỗi dữ liệu, và chỉ bật enhancement khi phép đo trên GT NavSim cho thấy có lợi.

Phạm vi hiện tại là camera `CAM_F0` của shard `2021.06.28.16.29.11_veh-38_01415_01821` (812 frame, 1920×1080). Các camera hoặc shard khác cần chạy quy trình kiểm tra tương tự. Không ghi đè ảnh nguồn; dữ liệu xử lý và kết quả đặt trong thư mục run riêng.

## Nguyên tắc

- Giữ nguyên ảnh gốc, thứ tự frame theo NavSim metadata, timestamp, calibration và đường dẫn ảnh.
- Không crop, kéo giãn riêng chiều rộng/chiều cao, cân bằng màu toàn cục hoặc làm nét mặc định. Các phép biến đổi này có thể đổi tỷ lệ/hình dạng người và làm sai phân phối ảnh mà model đã học.
- Dùng cùng preprocessing ở validation và inference. Mỗi thay đổi cần được so với baseline bằng projected NavSim GT, cùng shard, class, confidence và IoU threshold.
- Không xem kết quả inference trên shard dùng để chọn preprocessing là phép đo tổng quát hóa cuối cùng; xác nhận lựa chọn trên shard NavSim khác nếu có GT.

## Pipeline đề xuất

### 1. Kiểm kê và kiểm tra chất lượng đầu vào

- Đọc danh sách frame từ metadata NavSim, thay vì sắp thứ tự theo tên hash hoặc thời gian file.
- Kiểm tra ảnh mở được, số frame, kích thước, số kênh, kiểu dữ liệu, ảnh rỗng/trùng lặp và frame bị thiếu.
- Đảm bảo ảnh màu được đọc đúng thứ tự kênh mà Ultralytics mong đợi; không chuyển BGR/RGB hai lần.
- Ghi thống kê theo camera: kích thước, histogram độ sáng, tỷ lệ pixel tối/sáng, độ tương phản và chỉ số blur. Dùng các thống kê này để phát hiện nhóm frame bất thường, không tự động loại ảnh hợp lệ.
- Đối chiếu số frame và timestamp với metadata trước và sau pipeline; mọi frame bỏ qua phải được liệt kê kèm lý do.

### 2. Chuẩn hóa hình học không mất thông tin

- Giữ ảnh nguồn ở độ phân giải gốc, hiện tại 1920×1080.
- Để Ultralytics letterbox ảnh vào kích thước inference, giữ nguyên aspect ratio và padding theo stride. Không resize trực tiếp thành hình vuông và không lưu bản 640×640 làm đầu vào trung gian.
- Chạy baseline ở `imgsz=640` để so sánh công bằng với lần inference trước. Sau đó thử `imgsz=960` và `imgsz=1280` trên cùng dữ liệu. Kích thước lớn hơn giữ pedestrian nhỏ được nhiều pixel hơn, đổi lại tốn VRAM và thời gian.
- Nếu tăng `imgsz` chưa đủ cho pedestrian rất nhỏ, thử tiled inference trên ảnh gốc: chia thành tile có overlap, chạy detector trên từng tile, đưa box về tọa độ ảnh gốc rồi hợp nhất bằng class-aware NMS. Bắt đầu thử tile 960×960 với overlap 20–25%; loại bỏ tile rỗng chỉ khi không ảnh hưởng vùng biên và mọi box đều được quy đổi đúng.
- Ghi lại scale, padding, tile origin cho từng dự đoán; clip box sau khi quy đổi vào kích thước ảnh gốc. Kiểm tra trực quan các box gần mép tile để phát hiện box bị cắt hoặc ghép trùng.

### 3. Chuẩn hóa cường độ ảnh theo điều kiện thực tế

- Baseline giữ nguyên giá trị pixel. YOLO thực hiện chuẩn hóa tensor nội bộ; không tự chuẩn hóa z-score hoặc đổi mean/std bên ngoài.
- Tạo báo cáo exposure/contrast theo camera và theo frame. Nếu có nhóm frame tối hoặc tương phản thấp rõ rệt, thử riêng gamma nhẹ (ví dụ gamma 0.8–1.2) hoặc CLAHE trên kênh độ sáng trong không gian LAB.
- Giữ ảnh màu gốc; không áp CLAHE cho từng kênh màu và không áp enhancement mù trên toàn bộ shard.
- So sánh baseline với từng phép biến đổi riêng. Chỉ chọn enhancement nếu tăng recall/AP trên NavSim GT mà precision và chất lượng box không giảm quá mức. Khi chọn gamma/CLAHE, cố định tham số cho toàn bộ camera/shard thuộc cùng điều kiện; không tối ưu tham số dựa trên nhãn của từng frame.
- Không dùng sharpen/denoise mặc định: chúng có thể tạo cạnh giả quanh pedestrian nhỏ hoặc xóa chi tiết mảnh.

### 4. Inference và hậu xử lý

- Dùng checkpoint `best.pt`, class `pedestrian`, device và phiên bản Ultralytics cố định; ghi chúng vào report.
- Baseline inference dùng `imgsz=640`, confidence `0.25`, IoU/NMS theo cấu hình hiện tại. Khi so sánh preprocessing, cố định confidence và NMS; sau khi chọn pipeline mới có thể tune confidence trên một validation shard riêng.
- Tắt test-time augmentation trong baseline. Có thể đo TTA/flip riêng như một thí nghiệm; phải đảo tọa độ box về ảnh gốc và hợp nhất dự đoán trước khi tính metrics.
- Nếu dùng ByteTrack, luôn cung cấp frame theo metadata order và reset tracker giữa các shard. Detection metrics cần tính trên box detector sau cùng một quy tắc lọc; tracking metrics được báo riêng.
- Lưu ảnh visualize có label/confidence vào thư mục run, cùng prediction CSV/JSON, preprocessing config, metrics và danh sách frame lỗi. Không sửa ảnh gốc.

## Ma trận thí nghiệm

| Run | Hình học | Cường độ ảnh | Mục đích |
|---|---|---|---|
| A — baseline | Letterbox, 640 | Ảnh gốc | Mốc đối chiếu, khớp lần chạy hiện có |
| B | Letterbox, 960 | Ảnh gốc | Kiểm tra pedestrian nhỏ |
| C | Letterbox, 1280 | Ảnh gốc | Đo lợi ích và chi phí độ phân giải cao |
| D | Tile 960, overlap 20–25% | Ảnh gốc | Kiểm tra recall với object rất nhỏ |
| E — có điều kiện | Cấu hình tốt nhất từ B–D | Gamma hoặc CLAHE nhẹ, chỉ khi thống kê cho thấy vấn đề exposure | Kiểm tra adaptation ánh sáng |

Chạy mỗi cấu hình trên cùng frame và metadata. Bắt đầu bằng một tập đại diện gồm frame ban ngày/tối, cảnh đông/thưa, pedestrian gần/xa và frame khó; sau đó chạy toàn shard cho cấu hình đạt yêu cầu. Lưu toàn bộ cấu hình để có thể tái lập.

## Đánh giá và tiêu chí chọn

So sánh với projected GT NavSim ở IoU 0.5, cùng class mapping `pedestrian:pedestrian`. Báo cáo ít nhất:

- Precision, recall, F1, AP@0.5, mAP@0.5:0.95 và mean IoU của true positive.
- Recall/AP theo kích thước box (small/medium/large), độ sáng và nhóm camera nếu metadata cho phép.
- Số frame xử lý, lỗi decode, số detection, latency p50/p95, FPS, VRAM và dung lượng output.
- Với tile hoặc TTA: số box trước/sau merge và tỷ lệ box gần biên tile.

Chọn cấu hình theo mục tiêu ứng dụng đã thống nhất (ưu tiên recall pedestrian hay cân bằng precision/recall), đồng thời xem latency và VRAM. Không kết luận preprocessing tốt hơn chỉ từ ảnh minh họa hoặc confidence trung bình. Nếu mAP/recall tăng nhưng precision giảm, trình bày trade-off và chọn confidence trên validation độc lập.

## Kết quả hiện có làm baseline tham khảo

Lần chạy trước trên shard NavSim đã dùng `CAM_F0`, `imgsz=640`, confidence `0.25`, ByteTrack và 812 frame. So với projected GT pedestrian, kết quả là precision `0.1795`, recall `0.0688`, F1 `0.0995`, AP@0.5 `0.0179`. Đây là baseline cho shard đó; không so trực tiếp với mAP của bộ nuImages val vì tập GT và cách đánh giá khác nhau.

## Đầu ra dự kiến cho mỗi run

```text
<run_dir>/
├── preprocessing_navsim.md   # cấu hình thực tế và ghi chú run
├── evidence.md               # detection/tracking metrics và so sánh baseline
├── inference_metrics.json    # latency, throughput, frame count
├── predictions.csv           # box, class, confidence, frame/timestamp/track ID
└── visualized_frames/        # ảnh annotate; không chứa ảnh GT đã sửa
```

Không tạo bản sao ảnh “preprocessed” trừ khi phép biến đổi thực sự cần thiết. Nếu cần cache để tăng tốc, ghi cache vào thư mục run riêng, đặt tên theo camera và checksum/config, giữ nguyên ảnh nguồn, và đảm bảo inference có thể tái tạo từ cấu hình đã lưu.
