# CSP pedestrian inference trên NavSim mini

Project này dùng trực tiếp implementation từ [`wang-xinyu/csp.pytorch`](https://github.com/wang-xinyu/csp.pytorch) để chạy inference CSP ResNet-50, CityPersons Height + Offset trên ảnh local.

## Trạng thái theo Objectives.md

- Đã clone repository vào `csp.pytorch/`.
- Đã chuẩn bị script `inference_folder.py` để quét ảnh đệ quy, giữ nguyên cấu trúc thư mục camera, decode box bằng `parse_det_offset`, vẽ box + confidence và lưu vào `output/`.
- Dataset mặc định: `D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311`.
- Việc tải checkpoint và chạy thử một ảnh được để bạn tự thực hiện theo hướng dẫn bên dưới (đúng yêu cầu Side trong `Objectives.md`).

## 1. Môi trường

Máy kiểm tra có RTX 4050, driver báo CUDA 13.0 và Python 3.13.12. Tạo môi trường riêng trên Windows PowerShell:

```powershell
if (-not (Test-Path .\csp.pytorch)) { git clone https://github.com/wang-xinyu/csp.pytorch.git .\csp.pytorch }
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Kiểm tra CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu không dùng GPU, bỏ lệnh PyTorch CUDA và cài bản PyTorch phù hợp máy; script vẫn có thể chạy với `--device cpu` nhưng sẽ chậm.

## 2. Tải checkpoint `net_e121_l0.hdf5` khi Baidu khó tải

Baidu vẫn còn file gốc nhưng endpoint tải trực tiếp chặn phiên ẩn danh (`errno=9019`). Project có `download_baidu_checkpoint.py` để tự động xác thực link, lấy direct link, hỗ trợ file `.part` và kiểm tra MD5. Không gửi cookie Baidu qua chat và không commit cookie.

Checkpoint đã được xác nhận trên share gốc:

```text
remote path: /research/models_CSP/cityperson/withoffset/net_e121_l0.hdf5
size: 160609096 bytes
md5: 464fadd6a04527d00be91fe7769bdde1
share password: jcgd
```

Đăng nhập Baidu trong trình duyệt, lấy cookie `BDUSS` của `pan.baidu.com` từ DevTools, rồi chạy trong PowerShell. Cookie chỉ nằm trong cửa sổ PowerShell hiện tại:

```powershell
python -m pip install -r requirements.txt
$env:BAIDU_BDUSS = '<paste-BDUSS-locally-here>'
.\.venv\Scripts\python.exe .\download_baidu_checkpoint.py
Remove-Item Env:BAIDU_BDUSS
```

Nếu download bị ngắt, chạy lại lệnh; script sẽ tiếp tục từ `net_e121_l0.hdf5.part`. Nếu báo `errno=9019`, phiên đăng nhập hoặc cookie đã hết hạn, hãy đăng nhập lại và lấy lại `BDUSS`.

Repository CSP gốc cung cấp checkpoint qua Baidu Netdisk trong README của repo:

1. Mở [BaiduYun link của CSP](https://pan.baidu.com/s/1SSPQnbDP6zf9xf8eCDi3Fw).
2. Nhập mã: `jcgd`.
3. Lấy file `net_e121_l0.hdf5` tương ứng `cityperson/withoffset` (ResNet-50, Height + Offset).
4. Đặt file ở thư mục project, hoặc truyền đường dẫn tuyệt đối qua `--weights`.

Không dùng checkpoint face/widerface hoặc checkpoint Caltech `net_e82_l0.00850005054218.hdf5`: chúng không đúng biến thể CityPersons Height + Offset được yêu cầu.

Trước khi chạy folder, kiểm tra file có tồn tại:

```powershell
Get-Item .\net_e121_l0.hdf5
```

Loader gốc đọc Keras HDF5 layer names; không đổi sang `state_dict` PyTorch và không sửa kiến trúc model.

## 3. Tự chạy thử inference trên một ảnh

Chọn một ảnh trong dataset:

```powershell
$img = Get-ChildItem 'D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311' -Recurse -File -Include *.jpg,*.jpeg,*.png | Select-Object -First 1
$img.FullName
```

Chạy test chính thức của repo sau khi copy ảnh đó thành `shibuye.jpg` (test này tạo `out.jpg`):

```powershell
Copy-Item $img.FullName .\csp.pytorch\shibuye.jpg -Force
Copy-Item .\net_e121_l0.hdf5 .\csp.pytorch\net_e121_l0.hdf5 -Force
Push-Location .\csp.pytorch
..\.venv\Scripts\python.exe .\test_one_pic_pedestrian.py
Pop-Location
```

Lưu ý test gốc dùng ngưỡng `0.4` và chỉ vẽ box không ghi score. Script folder bên dưới dùng decoder gốc, ngưỡng CityPersons mặc định `0.1`, đồng thời ghi confidence.

## 4. Inference toàn bộ dataset

Sau khi đặt checkpoint ở root project:

```powershell
.\.venv\Scripts\python.exe .\inference_folder.py `
  --weights .\net_e121_l0.hdf5 `
  --input-dir 'D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311' `
  --output-dir .\output
```

Mỗi ảnh sẽ in số pedestrian phát hiện, ví dụ `pedestrians=3`. Kết quả giữ nguyên thư mục camera dưới `output/`, chẳng hạn `output\CAM_F0\<image>.jpg`.

Smoke test trên 2 ảnh trước khi chạy 6,672 ảnh:

```powershell
.\.venv\Scripts\python.exe .\inference_folder.py --weights .\net_e121_l0.hdf5 --limit 2
```

Tuỳ chọn hữu ích:

```text
--score-threshold 0.1   ngưỡng center confidence trước NMS
--device auto|cuda|cpu  tự chọn CUDA hoặc ép thiết bị
--limit N               chỉ xử lý N ảnh
```

## Preprocessing và kích thước ảnh

Script dùng nguyên `format_img` của repository CSP: ảnh BGR do OpenCV đọc, chuyển `float32`, trừ mean `[103.939, 116.779, 123.68]` theo từng kênh. CSP gốc cần kích thước chia hết cho stride 32 để ba nhánh `p3/p4/p5` ghép được; ảnh NavSim `1080×1920` được pad phía dưới thành `1088×1920` bằng channel mean, không resize và không sửa kiến trúc. Sau decode, box được clip và lọc lại theo kích thước gốc `1080×1920`. Decoder `parse_det_offset` dùng output stride `4`, `h = exp(height) * 4`, tỉ lệ pedestrian `w = 0.41 * h`, offset center và NMS IoU `0.5`.

## Giới hạn đã biết

- Checkpoint HDF5 không được commit vào repository vì kích thước lớn và bạn được yêu cầu tự tải. Nếu thiếu file, script dừng với thông báo rõ ràng.
- Repo gốc có các lệnh debug `print` trong residual blocks; script chỉ redirect log forward đó, không sửa model.
- Inference toàn bộ 6,672 ảnh có thể mất thời gian và tạo nhiều GB output; nên chạy `--limit 2` trước.
