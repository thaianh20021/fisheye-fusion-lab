# Tái dựng cảnh giao thông thực tế

## 1. Mục tiêu dự án

Tái dựng cảnh giao thông ngoài đời thực từ:

- Ảnh camera pinhole, có thể kết hợp LiDAR.
- Ảnh camera fisheye, có thể kết hợp LiDAR.

So sánh các phương pháp NeRF và 3D Gaussian Splatting (3DGS), xây dựng pipeline có thể tái lập, rồi đánh giá khả năng tích hợp kết quả vào CARLA.

## 2. Trạng thái nghiên cứu hiện tại

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| PandaSet sequence `001` | Sẵn sàng thử nghiệm | Có camera trước và LiDAR `Pandar64` |
| NeuRAD | Đã thử local | Cấu hình hiện tại quá nặng với RTX 3060 Laptop |
| SplatAD | Phương pháp chính | Đang chạy qua `neurad-studio` |
| Colab GPU | Đã chuẩn bị | Đã dùng GPU T4; cần xử lý lệch dependency |
| Thử nghiệm fisheye | Chưa hoàn thành | KITTI-360 là dataset tiếp theo |
| Tích hợp CARLA | Mới khảo sát | Chưa có kết quả import end-to-end |

## 3. Các phương pháp đang khảo sát

| Phương pháp | Nhóm | Cảm biến | Cảnh động | Vai trò |
|---|---|---|---|---|
| **SplatAD** | 3DGS | Pinhole, fisheye, LiDAR | Có | Phương pháp chính, render nhanh |
| **Street Gaussians** | 3DGS | Chủ yếu pinhole, LiDAR | Có | Tách xe và vật thể động |
| **Fisheye-GS** | 3DGS | Fisheye | Thường là cảnh tĩnh | Xử lý phép chiếu fisheye |
| **NeuRAD** | NeRF | Pinhole, fisheye, LiDAR | Có | Baseline NeRF |
| **EmerNeRF** | NeRF | Camera và tín hiệu hình học | Có | Tách nền tĩnh/vật thể động |

### SplatAD

Phù hợp nhất với mục tiêu vì hỗ trợ cảnh giao thông động, camera và LiDAR, đồng thời dùng 3DGS nên có tiềm năng render nhanh. Nhược điểm: tốn VRAM, phụ thuộc chặt vào phiên bản PyTorch/`gsplat`/NumPy/Numba, và huấn luyện có LiDAR nặng hơn camera-only.

Thí nghiệm hiện tại: PandaSet `001`, camera `front`, LiDAR `Pandar64`, chạy trên Colab T4 vì RTX 3060 Laptop quá chậm và nặng.

### Street Gaussians

Có ích khi cần tách nền đường tĩnh khỏi xe đang chạy, nhất là cho workflow CARLA có vật thể chỉnh sửa độc lập. Cần kiểm tra lại yêu cầu dataset, preprocessing và hỗ trợ fisheye.

### Fisheye-GS

Xử lý trực tiếp biến dạng ảnh góc rộng, phù hợp làm baseline cho nhánh fisheye. Cần xác minh khả năng xử lý cảnh động, LiDAR và xuất sang CARLA.

### NeuRAD

Là baseline NeRF phù hợp cho cảnh giao thông động và tạo đối chứng với SplatAD. Cấu hình hiện tại quá nặng trên RTX 3060 Laptop, nên dùng subset nhỏ hoặc GPU cloud.

### EmerNeRF

Phù hợp nghiên cứu tách thành phần tĩnh và động mà không phụ thuộc hoàn toàn vào 3D bounding box. Rủi ro: preprocessing/training lâu và khó xuất từng vật thể sang CARLA.

## 4. Các bộ dữ liệu

| Dataset | Pinhole | Fisheye | LiDAR | Vai trò |
|---|---:|---:|---:|---|
| **PandaSet** | Có | Không phải mục tiêu chính | Có | Smoke test pinhole nhanh |
| **KITTI-360** | Có | Có | Có | Dataset chính cho fisheye |
| **Waymo Open** | Có | Không phải mục tiêu chính | Có | Tham khảo cảnh lớn, phức tạp |
| **nuScenes** | Có | Không phải mục tiêu chính | Có | Dataset bổ sung |

### Thứ tự ưu tiên

1. **PandaSet `001`**: xác minh pipeline SplatAD nhanh.
2. **KITTI-360 đoạn ngắn**: thử camera fisheye và mô hình chiếu.
3. **Waymo/nuScenes**: chỉ dùng nếu còn thời gian hoặc phương pháp yêu cầu.

## 5. Pipeline đề xuất

### Pinhole + LiDAR

```text
PandaSet 001
  -> kiểm tra parser và calibration
  -> đồng bộ camera/LiDAR
  -> huấn luyện SplatAD hoặc NeuRAD
  -> render RGB/depth/LiDAR
  -> đánh giá chất lượng và tốc độ
```

### Fisheye + LiDAR

```text
KITTI-360
  -> kiểm tra mô hình camera fisheye
  -> chọn chiếu native hoặc baseline làm phẳng ảnh
  -> thử SplatAD / Fisheye-GS / NeuRAD
  -> render góc nhìn mới
  -> so sánh biến dạng rìa ảnh và hình học
```

## 6. Tiêu chí đánh giá

- **Hình ảnh:** PSNR, SSIM, LPIPS, kiểm tra trực quan xe động và rìa ảnh.
- **Hình học:** sai số depth, Chamfer distance, độ khớp với LiDAR, vùng bị thiếu.
- **Hiệu năng:** thời gian train, VRAM tối đa, kích thước checkpoint, FPS render, tỷ lệ lỗi.

## 7. Khả năng tích hợp CARLA

3DGS là ứng viên mạnh nhất cho render tương tác. SplatAD và Street Gaussians có tiềm năng tách vật thể động, nhưng cần xác minh output thực tế. Import trực tiếp vào CARLA chưa được chứng minh.

Hướng khả dĩ: xuất asset dạng point/mesh hoặc khảo sát workflow kiểu NVIDIA NuRec. Trước hết cần chuẩn hóa hệ tọa độ, calibration, timestamp và định dạng RGB/depth/LiDAR.

## 8. Việc cần làm ngay

1. Hoàn thành smoke test SplatAD một iteration trên PandaSet `001`.
2. Lưu checkpoint hợp lệ và output render đầu tiên.
3. Lưu dump phiên bản package của môi trường Colab.
4. Chỉ thêm KITTI-360 sau khi pipeline pinhole chạy ổn định.
5. Xem import CARLA là một thí nghiệm feasibility riêng, chưa coi là chắc chắn làm được.

