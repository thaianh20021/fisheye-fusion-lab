# Báo cáo kỹ thuật: Tái dựng cảnh giao thông thực tế bằng NeRF và 3D Gaussian Splatting

> **Dự án:** 3D Reconstruction Real-World  
> **Giai đoạn:** Phase 1 – Literature review & thiết lập môi trường (Tuần 1)  
> **Ngày cập nhật:** 07/11/2026  
> **Trạng thái:** Đang thực hiện – chưa có kết quả huấn luyện hoàn chỉnh

---

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Bối cảnh và mục tiêu](#2-bối-cảnh-và-mục-tiêu)
3. [Kiến thức nền tảng](#3-kiến-thức-nền-tảng)
4. [Khảo sát phương pháp](#4-khảo-sát-phương-pháp)
5. [Khảo sát bộ dữ liệu](#5-khảo-sát-bộ-dữ-liệu)
6. [Môi trường thực nghiệm](#6-môi-trường-thực-nghiệm)
7. [Nhật ký thực nghiệm và xử lý lỗi](#7-nhật-ký-thực-nghiệm-và-xử-lý-lỗi)
8. [Kết quả hiện tại](#8-kết-quả-hiện-tại)
9. [Pipeline đề xuất](#9-pipeline-đề-xuất)
10. [Phương pháp đánh giá](#10-phương-pháp-đánh-giá)
11. [Khả năng tích hợp CARLA](#11-khả-năng-tích-hợp-carla)
12. [Rủi ro và hướng giảm thiểu](#12-rủi-ro-và-hướng-giảm-thiểu)
13. [Kế hoạch tiếp theo](#13-kế-hoạch-tiếp-theo)
14. [Kết luận](#14-kết-luận)
15. [Tài liệu tham khảo](#15-tài-liệu-tham-khảo)
16. [Phụ lục](#16-phụ-lục)

---

## 1. Tóm tắt

Dự án khảo sát các phương pháp tái dựng cảnh 3D từ dữ liệu lái xe thực tế, gồm hai nhóm dữ liệu:

- Camera **pinhole** kết hợp LiDAR.
- Camera **fisheye** kết hợp LiDAR.

Sau Tuần 1, nhóm đã:

- Xác định 5 phương pháp ứng viên: **SplatAD, Street Gaussians, Fisheye-GS, NeuRAD, EmerNeRF**.
- Xác định 4 bộ dữ liệu: **PandaSet, KITTI-360, Waymo Open, nuScenes**.
- Chuẩn bị subset **PandaSet sequence `001`** (80 frame camera trước, LiDAR `Pandar64`).
- Chạy thử **NeuRAD** trên RTX 3060 Laptop: chạy được nhưng thời gian ước tính **khoảng 2 ngày**, không khả thi.
- Chuyển trọng tâm sang **SplatAD** (3DGS), thử trên máy cá nhân, Kaggle và Google Colab.
- Trên Colab T4, SplatAD đã qua bước import, đọc cấu hình và cache LiDAR, nhưng tiến trình bị dừng với `RETURN_CODE -9` (nghi do thiếu RAM).

**Chưa có checkpoint và chưa có số liệu đánh giá.** Việc trước mắt là hoàn thành smoke test SplatAD với cấu hình tối thiểu.

---

## 2. Bối cảnh và mục tiêu

### 2.1 Bối cảnh

Mô phỏng lái xe tự động cần môi trường giống thực tế. Simulator như CARLA dùng asset dựng thủ công nên có khoảng cách với thực tế (sim-to-real gap). Neural rendering cho phép tái dựng trực tiếp cảnh thật từ log cảm biến rồi render lại từ góc nhìn mới.

### 2.2 Mục tiêu

| Loại dữ liệu | Cảm biến | Đặc điểm |
|---|---|---|
| Pinhole | Camera thường + LiDAR | Tia sáng thẳng, phổ biến, nhiều công cụ hỗ trợ |
| Fisheye | Camera mắt cá + LiDAR | Góc rộng (~180°), biến dạng mạnh ở rìa ảnh |

### 2.3 Deliverables

1. Pipeline tái dựng cho mỗi phương pháp triển vọng.
2. Một cảnh đã tái dựng cho mỗi phương pháp triển vọng.
3. Nghiên cứu khả năng tích hợp CARLA.
4. Báo cáo đánh giá kỹ thuật.

### 2.4 Kế hoạch 6 tuần

| Phase | Thời gian | Nội dung |
|---|---|---|
| Phase 1 | Tuần 1 | Literature review, chọn method và dataset, cài đặt môi trường |
| Phase 2 | Tuần 2–4 | Mỗi thành viên phụ trách một method, chạy thực nghiệm |
| Phase 3 | Tuần 5–6 | Gom về pipeline chung, đánh giá, khảo sát CARLA, viết báo cáo |

---

## 3. Kiến thức nền tảng

### 3.1 NeRF (Neural Radiance Fields)

- Biểu diễn cảnh bằng mạng neural ánh xạ `(vị trí 3D, hướng nhìn) → (màu, mật độ)`.
- Render bằng cách bắn tia qua từng pixel và lấy mẫu nhiều điểm dọc tia (volume rendering).
- **Ưu điểm:** chất lượng cao, biểu diễn liên tục, mô phỏng tia LiDAR tự nhiên.
- **Nhược điểm:** huấn luyện và render chậm, khó chạy thời gian thực.

### 3.2 3D Gaussian Splatting (3DGS)

- Biểu diễn cảnh bằng rất nhiều Gaussian 3D, mỗi Gaussian có vị trí, hình dạng, độ trong suốt và màu.
- Render bằng cách chiếu Gaussian lên mặt phẳng ảnh và trộn màu theo độ sâu (rasterization).
- **Ưu điểm:** render nhanh (có thể thời gian thực), huấn luyện nhanh hơn NeRF.
- **Nhược điểm:** tốn bộ nhớ; phép chiếu gốc giả định camera pinhole.

### 3.3 So sánh

| Tiêu chí | NeRF | 3DGS |
|---|---|---|
| Biểu diễn | Ngầm (mạng neural) | Tường minh (tập Gaussian) |
| Tốc độ render | Chậm | Nhanh |
| Tốc độ huấn luyện | Chậm | Nhanh hơn |
| Bộ nhớ | Thấp hơn | Cao hơn |
| Camera fisheye | Dễ (đổi hàm sinh tia) | Khó hơn (phải sửa phép chiếu) |
| LiDAR | Tự nhiên (ray casting) | Cần rasterizer riêng |
| Phù hợp CARLA | Thấp (chậm) | Cao hơn |

### 3.4 Thách thức của cảnh lái xe

- **Cảnh động:** xe, người di chuyển; cần tách nền tĩnh và vật thể động.
- **Quỹ đạo gần tuyến tính:** ít góc nhìn, dễ overfit.
- **Đa cảm biến:** cần đồng bộ timestamp và calibration camera–LiDAR.
- **Rolling shutter:** cảm biến quét theo thời gian khi xe đang chạy.
- **Fisheye:** biến dạng mạnh ở rìa, không tương thích trực tiếp với 3DGS chuẩn.

---

## 4. Khảo sát phương pháp

### 4.1 Tổng quan

| Phương pháp | Nhóm | Camera | Khởi tạo / depth | Biểu diễn | Trường hợp phù hợp | Điểm mạnh | Hạn chế chính | Mức phù hợp dự án |
|---|---|---|---|---|---|---|---|---|
| **SplatAD** | 3DGS | Pinhole + rolling shutter | Có; mô phỏng LiDAR | Gaussian scene graph, camera + LiDAR | Camera + LiDAR, cảnh giao thông động | Render camera và LiDAR thời gian thực; phù hợp autonomous driving | Cần custom `gsplat`; phụ thuộc 3D box; người thường biểu diễn như rigid box | **Ưu tiên 1** |
| **NeuRAD** | NeRF | Pinhole; fisheye cần xác minh | Có | Neural field cho cảnh động | Baseline NeRF camera + LiDAR | Mô hình hóa cảnh động, rolling shutter và LiDAR | Huấn luyện rất nặng trên RTX 3060 Laptop | **Ưu tiên 2** |
| **Street Gaussians** | 3DGS | Pinhole | Init + depth loss | Gaussian scene graph cho nền và xe | Cảnh đô thị nhiều xe | Tách xe thành asset riêng; thuận lợi cho chỉnh sửa scene | Chủ yếu xe; cần 3D box; không có fisheye native | **Ưu tiên 3** |
| **OmniRe** | 3DGS | Pinhole | Init + depth loss | Gaussian scene graph, có SMPL người | Cảnh có người đi bộ và cyclist | Mô hình hóa nhiều actor; hỗ trợ 6 dataset; actor có thể chỉnh sửa | Cần SMPL, human pose và preprocessing box | Ưu tiên 4 nếu cần người |
| **3DGUT** | 3DGS | Pinhole, fisheye, rolling shutter | Không mô hình hóa LiDAR | Static Gaussians | Baseline fisheye tĩnh | Native fisheye; tốc độ render cao; liên quan workflow NVIDIA NuRec | Chỉ cảnh tĩnh; không xử lý actor động | **Ưu tiên fisheye baseline** |
| **IDSplat** | 3DGS | Pinhole | Instance-level losses | Instance-level Gaussian scene | Dữ liệu thiếu 3D box | Không cần nhãn 3D đầy đủ; kết quả tương đương SplatAD theo công bố | Chủ yếu actor rigid; cần kiểm tra pipeline thực tế | Ưu tiên 5 |
| **STORM** | Feed-forward | Pinhole | Trong giai đoạn train | Gaussians + velocity | Chạy nhiều scene nhanh | Khoảng 0,2 giây/scene theo báo cáo; không cần box | Không có official weights; độ phân giải thấp; cần train model | Tham khảo mở rộng |
| **DGGT** | Feed-forward | Pinhole, unposed | Trong giai đoạn train | Gaussians + pose + motion | Không có calibration | Không cần calibration; zero-shot nhanh | Chất lượng mỗi scene thấp hơn; checkpoint hạn chế | Tham khảo mở rộng |
| **Instant NuRec** | Feed-forward | Fisheye qua 3DGUT và camera khác | Training với depth | Gaussian phân lớp: static, dynamic, sky | Production-scale simulation | Khoảng 1,5 giây/scene; hệ sinh thái NVIDIA; native fisheye | Chất lượng thấp hơn NuRec per-scene khoảng 4,5 dB theo bảng khảo sát | **Ưu tiên CARLA/NVIDIA** |
| **UniGaussian** | Fisheye | Pinhole + fisheye, gồm KITTI-360 | Init + depth; có thể mô phỏng LiDAR | Composite Gaussians có vật thể động | Cảnh lái xe fisheye + LiDAR | Hướng công bố trực tiếp cho dynamic fisheye driving | Chưa có code công khai; khó tái lập | Ưu tiên literature |
| **GOFish** | Fisheye | Fisheye đến hơn 200 độ | Không dùng LiDAR | Static Gaussians + mesh | Fisheye rất rộng, cần xuất mesh | Giữ field of view rộng; có mesh export | Cảnh tĩnh; chưa kiểm chứng trên driving data | Tham khảo |
| **DirectFisheye-GS** | Fisheye | Fisheye, Kannala–Brandt | Không | Static Gaussians | Fisheye có calibration thực | Xử lý calibration thật; biên ảnh tốt hơn 3DGUT theo báo cáo | Chưa tìm thấy code hoàn chỉnh; cảnh tĩnh | Tham khảo |
| **Fisheye-GS** | Fisheye | Fisheye, equidistant | Không | Static Gaussians | Baseline fisheye nhanh | Module drop-in đơn giản; có code | Chỉ equidistant; cảnh tĩnh; bị 3DGUT vượt qua trong một số so sánh | **Ưu tiên fisheye baseline** |

> **Mức độ tin cậy:** bảng này là bản tổng hợp nghiên cứu ban đầu. Các con số tốc độ, chất lượng và khả năng hỗ trợ cảm biến phải được kiểm tra lại từ paper/repository trước khi dùng làm kết luận định lượng. "Cần xác minh" nghĩa là nhóm chưa chạy trên dữ liệu thực tế.

### 4.2 Phân nhóm theo mục tiêu nghiên cứu

#### Nhóm A – Cảnh lái xe động, camera + LiDAR

- **SplatAD:** ứng viên chính vì khớp trực tiếp với PandaSet và mục tiêu render camera/LiDAR.
- **NeuRAD:** baseline NeRF để so sánh chất lượng và chi phí tính toán.
- **Street Gaussians:** ứng viên khi cần tách xe thành các Gaussian object có thể chỉnh sửa.
- **OmniRe:** đáng xem xét nếu nhóm muốn đánh giá người đi bộ/cyclist, nhưng preprocessing phức tạp hơn.
- **IDSplat:** hướng thay thế khi thiếu nhãn 3D box.

#### Nhóm B – Fisheye và rolling shutter

- **3DGUT:** baseline quan trọng cho native fisheye và rolling shutter ở cảnh tĩnh.
- **Fisheye-GS:** baseline dễ triển khai, phù hợp kiểm tra ảnh đã calibration hoặc equidistant.
- **DirectFisheye-GS:** tham khảo thêm cho Kannala–Brandt.
- **UniGaussian:** hướng nghiên cứu gần nhất với mục tiêu dynamic fisheye + LiDAR, nhưng khả năng tái lập cần xác minh.

#### Nhóm C – Feed-forward và mô phỏng quy mô lớn

- **STORM:** phù hợp khi cần tái dựng rất nhiều scene với thời gian mỗi scene cực ngắn.
- **DGGT:** phù hợp khi pose hoặc calibration không đầy đủ.
- **Instant NuRec:** ứng viên khảo sát cho pipeline sản xuất và khả năng kết nối hệ sinh thái NVIDIA/CARLA.

### 4.3 Phân tích từng phương pháp bổ sung

#### OmniRe

OmniRe mở rộng biểu diễn Gaussian scene graph cho nhiều actor, trong đó con người được biểu diễn bằng mô hình SMPL và pose theo thời gian. Phương pháp này phù hợp với cảnh đô thị có người đi bộ hoặc cyclist, nhưng yêu cầu preprocessing human pose và dữ liệu actor rõ ràng. Không nên chọn làm smoke test đầu tiên vì chi phí chuẩn bị dữ liệu cao hơn SplatAD.

#### 3DGUT

3DGUT là baseline quan trọng cho camera không pinhole, gồm fisheye và rolling shutter. Phương pháp này phù hợp để trả lời câu hỏi "3DGS chuẩn bị sai ở đâu khi camera fisheye?" Tuy nhiên, vì trọng tâm là static scene, cần tách riêng kết luận về phép chiếu khỏi kết luận về cảnh giao thông động.

#### IDSplat

IDSplat dùng thông tin instance và các loss liên quan để học Gaussian scene mà không phụ thuộc hoàn toàn vào 3D bounding box. Đây là hướng đáng thử nếu PandaSet hoặc dataset phụ có nhãn box không đầy đủ. Cần kiểm tra khả năng xuất actor và khả năng dùng chung format với pipeline đánh giá.

#### STORM và DGGT

Hai phương pháp feed-forward ưu tiên tốc độ và khả năng mở rộng hơn chất lượng tối đa từng scene. Chúng phù hợp làm nhánh nghiên cứu bổ sung nếu dự án cần đánh giá throughput hoặc trường hợp không có calibration. Không nên đưa vào deliverable chính trước khi pipeline SplatAD chạy ổn định.

#### Instant NuRec

Instant NuRec đáng khảo sát riêng cho phần CARLA/NVIDIA vì sử dụng biểu diễn Gaussian phân lớp và hướng tới production-scale simulation. Cần phân biệt rõ **NuRec per-scene optimization** với **Instant NuRec feed-forward**; hai hướng có trade-off khác nhau về thời gian chuẩn bị và chất lượng.

#### UniGaussian, GOFish và DirectFisheye-GS

Đây là nhóm tham khảo cho nhánh fisheye. UniGaussian gần với bài toán driving động nhất nhưng chưa có code công khai theo thông tin hiện tại. GOFish và DirectFisheye-GS hữu ích để so sánh mô hình camera, nhưng chủ yếu là static scene và không thay thế được pipeline dynamic + LiDAR.

### 4.4 SplatAD

- **Nguồn:** Hess et al., CVPR 2025; mã nguồn trong `neurad-studio`.
- **Ý tưởng:** 3DGS render đồng thời camera và LiDAR cho cảnh lái xe; mô hình hóa rolling shutter, cường độ LiDAR và ray drop.
- **Lý do chọn:** khớp nhất với yêu cầu (camera + LiDAR + cảnh động + render nhanh).
- **Yêu cầu:** `gsplat` bản có kernel LiDAR (`map_points_to_lidar_tiles`, …), GPU NVIDIA, 3D bounding box cho vật thể động.
- **Rủi ro:** phụ thuộc chặt phiên bản; cần biên dịch CUDA; tốn RAM/VRAM khi cache dữ liệu.

### 4.5 Street Gaussians

- **Nguồn:** Yan et al., ECCV 2024.
- **Ý tưởng:** nền là một tập Gaussian tĩnh; mỗi xe là một tập Gaussian riêng gắn với pose theo thời gian.
- **Ưu điểm:** dễ chỉnh sửa, di chuyển, xóa từng xe; hợp với sinh kịch bản mô phỏng.
- **Hạn chế:** không có fisheye native; LiDAR chủ yếu để khởi tạo; codebase riêng.

### 4.6 Fisheye-GS

- **Nguồn:** Liao et al., 2024.
- **Ý tưởng:** sửa phép chiếu và đạo hàm của 3DGS theo mô hình camera equidistant để huấn luyện trực tiếp trên ảnh fisheye.
- **Vai trò:** đối chứng hai hướng xử lý fisheye:
  - **Hướng A:** làm phẳng ảnh (undistort) rồi dùng 3DGS chuẩn – đơn giản nhưng mất vùng rìa.
  - **Hướng B:** chiếu native – giữ trọn trường nhìn nhưng phải sửa rasterizer.
- **Hạn chế:** chủ yếu cảnh tĩnh, không render LiDAR.

### 4.7 NeuRAD

- **Nguồn:** Tonderski et al., CVPR 2024; cùng codebase `neurad-studio`.
- **Ý tưởng:** NeRF cho cảnh lái xe, render camera và LiDAR, mô hình hóa rolling shutter và vật thể động.
- **Vai trò:** baseline NeRF so sánh công bằng với SplatAD (cùng dữ liệu, cùng codebase).
- **Kết quả thử:** chạy được trên RTX 3060 Laptop nhưng ETA 1,7–2,7 ngày (Mục 7).

### 4.8 EmerNeRF

- **Nguồn:** Yang et al., ICLR 2024.
- **Ý tưởng:** tự tách trường tĩnh, trường động và trường luồng bằng self-supervision, không cần 3D bounding box.
- **Vai trò:** tham khảo khi dataset thiếu nhãn vật thể.
- **Hạn chế:** chậm; khó xuất từng vật thể sang môi trường mô phỏng.

### 4.9 Mức ưu tiên

| Ưu tiên | Phương pháp | Lý do |
|---|---|---|
| 1 | SplatAD | Đáp ứng nhiều yêu cầu nhất, render nhanh, đã có môi trường |
| 2 | NeuRAD | Baseline cùng codebase, đã chạy được |
| 3 | Fisheye-GS | Cần cho nhánh fisheye |
| 4 | Street Gaussians | Quan trọng nếu cần chỉnh sửa vật thể trong CARLA |
| 5 | EmerNeRF | Tham khảo, chạy nếu còn thời gian |

---

## 5. Khảo sát bộ dữ liệu

### 5.1 Tổng quan

| Dataset | Camera | Fisheye | LiDAR | 3D box | Vai trò |
|---|---|:---:|---|:---:|---|
| **PandaSet** | 6 camera | ✗ | Pandar64 + PandarGT | ✓ | Smoke test pinhole |
| **KITTI-360** | 2 perspective + 2 fisheye | ✓ | Velodyne 64-beam | ✓ | Dataset chính cho fisheye |
| **Waymo Open** | 5 camera | ✗ | 5 LiDAR | ✓ | Cảnh phức tạp, tham khảo |
| **nuScenes** | 6 camera | ✗ | 32-beam | ✓ | Tham khảo bổ sung |

### 5.2 PandaSet (đang dùng)

- Mỗi sequence khoảng 8 giây, 10 Hz (~80 frame).
- `neurad-studio` hỗ trợ sẵn qua `pandaset-data`.
- **Subset hiện có:** sequence `001`, camera `front`, LiDAR `Pandar64`, 80 ảnh, 80 timestamp.
- **Giới hạn:** không có fisheye.

### 5.3 KITTI-360 (tiếp theo)

- Có camera perspective, hai camera fisheye bên hông và LiDAR trong cùng hệ tọa độ.
- **Rủi ro:** dung lượng lớn nên chỉ tải đoạn ngắn; cần kiểm tra parser trong `neurad-studio`.

### 5.4 Thứ tự sử dụng

1. PandaSet `001` – xác minh pipeline pinhole.
2. KITTI-360 (đoạn ngắn) – nhánh fisheye.
3. Waymo / nuScenes – chỉ khi còn thời gian.

---

## 6. Môi trường thực nghiệm

| Môi trường | GPU | Trạng thái | Ghi chú |
|---|---|---|---|
| Laptop cá nhân (Windows) | RTX 3060 Laptop 6 GB | Chạy được NeuRAD, quá chậm | Windows không hỗ trợ `torch.compile`; thiếu Ninja cho `gsplat` |
| Kaggle | 2× Tesla T4 16 GB | Tạm bỏ | Python 3.13 mặc định, xung đột dependency, tạo venv Python 3.10 thất bại |
| Google Colab | Tesla T4 16 GB | Đang dùng | Cài được SplatAD; giới hạn RAM hệ thống |

**Codebase:** `neurad-studio` (NeuRAD + SplatAD, xây trên nerfstudio).

---

## 7. Nhật ký thực nghiệm và xử lý lỗi

### 7.1 NeuRAD trên RTX 3060 Laptop

```text
neurad, num_processes=0, train/eval rays per batch = 1024
pandaset-data, sequence 001, dataset_end_fraction=0.2, cameras front
```

| Step | Thời gian/iter | ETA | Rays/s |
|---|---|---|---|
| 0 | 7,2 s | 1 ngày 16 giờ | – |
| 100 | 11,7 s | 2 ngày 16 giờ | 93 |
| 200 | 8,7 s | 1 ngày 23 giờ | 120 |

**Kết luận:** không khả thi trên laptop. Đã dừng và chuyển sang SplatAD trên cloud.

### 7.2 SplatAD – các lỗi đã gặp

| # | Môi trường | Lỗi | Nguyên nhân | Xử lý |
|---|---|---|---|---|
| 1 | Local | `Unrecognized options: --pipeline.datamanager.num_processes` | CLI dùng dấu gạch ngang | Đổi thành `--pipeline.datamanager.num-processes` |
| 2 | Local | `RuntimeError: Ninja is required to load C++ extensions` | `gsplat` biên dịch CUDA lúc chạy | Cần Ninja + MSVC; chuyển sang cloud |
| 3 | Kaggle | `Failed to build 'numba'` | Python 3.13 không hợp `numba==0.57` | Thử venv Python 3.10 |
| 4 | Kaggle | `No module named 'av2'` | Thiếu dependency dataparser | Cài `av2` |
| 5 | Kaggle | `uv venv` thất bại | Sai đường dẫn `uv`, venv đã tồn tại | Chuyển sang Colab |
| 6 | Colab | `cannot import name 'JSONSerializable' from 'dataclass_wizard'` | `dataclass_wizard` mới không hợp `zod` | Hạ phiên bản `dataclass_wizard` |
| 7 | Colab | `cannot import name 'map_points_to_lidar_tiles' from 'gsplat'` | Dùng `gsplat` gốc, thiếu kernel LiDAR | Cài `gsplat` theo hướng dẫn `neurad-studio` |
| 8 | Colab | `Numba needs NumPy 2.0 or less. Got NumPy 2.5` | NumPy quá mới | Cài `numpy<2` |
| 9 | Colab | `operator torchvision::nms does not exist` | `torch` và `torchvision` lệch phiên bản | Cài cặp phiên bản khớp nhau |
| 10 | Colab | `Unrecognized or misplaced options` | Tham số đặt sai vị trí subcommand | Đặt tham số `pipeline` trước `pandaset-data` |
| 11 | Colab | `np.bool8` không tồn tại | NumPy mới bỏ alias | Patch thành `np.bool_` |
| 12 | Colab | Lỗi gán vận tốc có mask trong `ad_dataparser.py` | Không tương thích phiên bản thư viện | Patch 2 phép gán |
| 13 | Colab | `RETURN_CODE -9` sau `Caching train lidars` | Nghi thiếu RAM khi cache | Giảm dữ liệu và độ phân giải (Mục 13.1) |

### 7.3 Bài học

- `neurad-studio` cần đúng bộ phiên bản: Python 3.10, Torch CUDA 11.8, `numpy<2`, `gsplat` bản có kernel LiDAR.
- Python 3.12/3.13 mặc định của Colab/Kaggle gây nhiều xung đột.
- Lưu `pip freeze` ngay khi môi trường chạy được.
- Bắt đầu bằng cấu hình rất nhỏ (1 iteration) trước khi chạy dài.

---

## 8. Kết quả hiện tại

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Dataset PandaSet `001` | ✅ Sẵn sàng | 80 ảnh camera trước, 80 timestamp, LiDAR `Pandar64` |
| NeuRAD | ⚠️ Chạy nhưng quá chậm | Log ETA ~2 ngày trên RTX 3060 Laptop |
| SplatAD import | ✅ Thành công | Import không lỗi trên Colab |
| SplatAD đọc cấu hình | ✅ Thành công | Lưu được `config.yml` |
| SplatAD cache LiDAR | ✅ Thành công | Log `Caching train lidars` |
| SplatAD huấn luyện | ❌ Chưa | Dừng với `RETURN_CODE -9` |
| Instant NuRec inference | ✅ Thành công | 2 chunks; xuất 1.922.999 static và 75.730 dynamic Gaussians |
| Checkpoint | ❌ Chưa có | – |
| Metric (PSNR/SSIM/LPIPS) | ❌ Chưa có | – |
| Fisheye | ❌ Chưa bắt đầu | – |
| CARLA | ❌ Chưa thử | – |

### 8.1 Kết quả Instant NuRec

Instant NuRec đã chạy thành công trên Colab bằng pretrained checkpoint `instant_nurec_pa_front_1.1.0.pth`.

```text
Predicting: 2/2 chunks
Merging: 3.272.492 Gaussians
Static sau voxelization: 1.922.999 Gaussians
Dynamic layer: 75.730 Gaussians, 3 keyframes
```

Các file đã xuất:

- `.ply`: scene tĩnh gồm 1.922.999 Gaussian.
- `.dynamic.npz`: layer động gồm 75.730 Gaussian.
- `.sky.npz`: sky sidecar và camera metadata.
- `.sky.png`: ảnh preview sky.
- `.render.png`: ảnh render đã composite với sky; background chiếm 67,4%, sky contribution `0.2723`.

Kết quả này xác nhận pipeline **NCore → Instant NuRec inference → Gaussian scene export → sky-composited render** hoạt động. Đây là kết quả inference bằng model pretrained, chưa phải kết quả train lại trên PandaSet `001`.

**Output preview hiện tại:**

```text
/content/instant-nurec/demo_output_preview/2aGVfhjSknwYQCUXbm6UcF/ply/
pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb/
```

---

## 9. Pipeline đề xuất

### 9.1 Pipeline A – Pinhole + LiDAR

```text
PandaSet 001
  → kiểm tra parser và calibration
  → đồng bộ camera/LiDAR theo timestamp
  → huấn luyện SplatAD (chính) và NeuRAD (baseline)
  → render RGB / depth / LiDAR trên frame giữ lại
  → tính metric, đo thời gian và VRAM
```

### 9.2 Pipeline B – Fisheye + LiDAR

```text
KITTI-360 (đoạn ngắn)
  → kiểm tra mô hình camera fisheye
  → nhánh 1: undistort ảnh → SplatAD / 3DGS chuẩn
  → nhánh 2: chiếu native → Fisheye-GS / SplatAD (nếu hỗ trợ)
  → render góc nhìn mới
  → so sánh biến dạng rìa ảnh và độ chính xác hình học
```

### 9.3 Định dạng đầu ra chung

- Ảnh RGB render ở cùng frame, cùng độ phân giải.
- Depth map (nếu có).
- Point cloud LiDAR render (nếu có).
- Pose camera trong cùng hệ tọa độ.
- Log thời gian huấn luyện và VRAM tối đa.

---

## 10. Phương pháp đánh giá

### 10.1 Chia dữ liệu

- Giữ lại mỗi frame thứ N (ví dụ N = 8) làm tập kiểm tra, không dùng khi huấn luyện.
- Mọi phương pháp dùng cùng một cách chia.

### 10.2 Metric

| Nhóm | Metric | Ý nghĩa |
|---|---|---|
| Hình ảnh | PSNR ↑ | Sai số pixel |
| Hình ảnh | SSIM ↑ | Tương đồng cấu trúc |
| Hình ảnh | LPIPS ↓ | Tương đồng cảm nhận |
| LiDAR | Depth L1 / median error ↓ | Sai số khoảng cách |
| LiDAR | Chamfer distance ↓ | Độ khớp point cloud |
| LiDAR | Ray drop accuracy ↑ | Dự đoán điểm bị mất |
| Hiệu năng | Thời gian huấn luyện ↓ | Chi phí |
| Hiệu năng | VRAM tối đa ↓ | Khả năng chạy trên phần cứng hạn chế |
| Hiệu năng | FPS render ↑ | Khả năng dùng trong simulator |

### 10.3 Đánh giá định tính

- Chất lượng xe đang di chuyển (mờ, bóng ma, mất chi tiết).
- Vùng rìa ảnh fisheye.
- Render ở góc lệch khỏi quỹ đạo gốc (ví dụ dịch ngang 1–3 m).

---

## 11. Khả năng tích hợp CARLA

### 11.1 Câu hỏi cần trả lời

| Câu hỏi | Đánh giá hiện tại |
|---|---|
| Có render tương tác được không? | 3DGS là ứng viên mạnh nhất; NeRF quá chậm |
| Có tách được vật thể động không? | SplatAD và Street Gaussians có tiềm năng; cần kiểm tra output |
| Có import trực tiếp vào CARLA không? | Chưa chứng minh |
| Cần chuẩn hóa gì? | Hệ tọa độ, calibration, timestamp, định dạng RGB/depth/LiDAR |

### 11.2 Các hướng khả dĩ

| Hướng | Mô tả | Ưu điểm | Nhược điểm |
|---|---|---|---|
| A. NVIDIA NuRec | Dùng công cụ neural reconstruction của NVIDIA kết hợp CARLA | Hướng chính thức, render 3DGS | Cần kiểm tra phiên bản, phần cứng, định dạng dữ liệu |
| B. Xuất mesh | Chuyển cảnh tái dựng thành mesh, import như asset | Tương thích Unreal/CARLA | Mất chất lượng và hiệu ứng phụ thuộc góc nhìn |
| C. Render ngoài | CARLA điều khiển xe/pose; renderer 3DGS sinh ảnh bên ngoài | Giữ chất lượng | Phải đồng bộ hai hệ thống, có độ trễ |

### 11.3 Việc cần làm

1. Xác minh phiên bản CARLA và NuRec được hỗ trợ, cùng định dạng scene đầu vào.
2. Thử chuyển output SplatAD sang định dạng đó.
3. Đo FPS khi render trong vòng lặp mô phỏng.

---

## 12. Rủi ro và hướng giảm thiểu

| Rủi ro | Mức độ | Hướng giảm thiểu |
|---|---|---|
| GPU cá nhân yếu | Cao | Dùng Colab/Kaggle; cấu hình nhỏ; xin GPU từ dự án |
| Xung đột dependency | Cao | Cố định Python 3.10, lưu `pip freeze`, dùng Docker nếu được |
| Colab giới hạn RAM và thời gian | Cao | Giảm subset; lưu checkpoint thường xuyên lên Drive |
| Dataset fisheye lớn | Trung bình | Chỉ tải đoạn ngắn KITTI-360 |
| Fisheye không hỗ trợ native | Trung bình | Làm baseline undistort trước |
| CARLA không import được | Trung bình | Báo cáo feasibility, đề xuất hướng thay thế |
| 6 tuần ngắn | Cao | Ưu tiên SplatAD + NeuRAD; method khác là tùy chọn |

---

## 13. Kế hoạch tiếp theo

### 13.1 Ngay lập tức – Smoke test SplatAD

```bash
cd /content/neurad-studio
PYTHONPATH=/content CUDA_VISIBLE_DEVICES=0 python -u nerfstudio/scripts/train.py splatad \
  --vis tensorboard \
  --max-num-iterations 1 \
  --steps-per-save 1 \
  --pipeline.datamanager.camera-res-scale-factor 0.0625 \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.datamanager.cache-lidars cpu \
  --pipeline.model.max-num-seed-points 200 \
  pandaset-data \
  --data /content/neurad-studio/data/pandaset \
  --sequence 001 \
  --dataset-start-fraction 0.0 \
  --dataset-end-fraction 0.005 \
  --cameras front \
  --lidars Pandar64
```

**Điều kiện thành công:**

- Log có dòng `Step (...)`.
- Tiến trình kết thúc với mã `0`.
- Có file `.ckpt` trong `outputs/`.

**Sau đó:** tăng dần `dataset-end-fraction` (0,01 → 0,02 → 0,2), số iteration (20 → 1.000 → 30.000) và độ phân giải.

### 13.2 Tuần 2–4

- [ ] Huấn luyện SplatAD đầy đủ trên PandaSet `001`.
- [ ] Chạy NeuRAD trên cùng subset bằng GPU cloud.
- [ ] Tính metric, so sánh hai phương pháp.
- [ ] Tải đoạn ngắn KITTI-360, thử nhánh fisheye.
- [ ] Thử Fisheye-GS và baseline undistort.

### 13.3 Tuần 5–6

- [ ] Chuẩn hóa output, viết script đánh giá chung.
- [ ] Khảo sát NuRec / CARLA.
- [ ] Hoàn thiện báo cáo kỹ thuật.

---

## 14. Kết luận

- **3DGS, cụ thể SplatAD,** là hướng phù hợp nhất vì hỗ trợ camera, LiDAR, cảnh động và render nhanh – điều kiện cần cho tích hợp CARLA.
- **NeuRAD** là baseline NeRF; đã chạy được nhưng cần GPU mạnh hơn.
- **Fisheye** chưa được kiểm chứng và có rủi ro cao nhất; cần KITTI-360 và so sánh undistort với chiếu native.
- **Rào cản chính hiện nay là môi trường** (phần cứng, dependency), không phải lý thuyết. Ưu tiên số 1 là có checkpoint SplatAD đầu tiên.

---

## 15. Tài liệu tham khảo

> Kiểm tra lại các liên kết trước khi đưa vào báo cáo chính thức.

**Phương pháp**

1. Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023. https://arxiv.org/abs/2308.04079
2. Hess et al., *SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving*, CVPR 2025. https://arxiv.org/abs/2411.16816
3. Tonderski et al., *NeuRAD: Neural Rendering for Autonomous Driving*, CVPR 2024. https://arxiv.org/abs/2311.15260
4. Yan et al., *Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting*, ECCV 2024. https://arxiv.org/abs/2401.01339
5. Yang et al., *EmerNeRF: Emergent Spatial-Temporal Scene Decomposition via Self-Supervision*, ICLR 2024. https://arxiv.org/abs/2311.02077
6. Liao et al., *Fisheye-GS: Lightweight and Extensible Gaussian Splatting Module for Fisheye Cameras*, 2024. https://arxiv.org/abs/2409.04751
7. *OmniRe*, mã nguồn và tài liệu: https://github.com/ziyc/drivestudio
8. *3DGUT*, mã nguồn NVIDIA: https://github.com/nv-tlabs/3dgrut
9. *IDSplat*, mã nguồn: https://github.com/zenseact/idsplat
10. *GaussianSTORM*, mã nguồn NVIDIA: https://github.com/NVlabs/GaussianSTORM
11. *DGGT*, mã nguồn: https://github.com/xiaomi-research/dggt
12. *Instant NuRec*, trang dự án NVIDIA: https://research.nvidia.com/labs/sil/projects/instant-nurec/
13. *GOFish*: https://3dgofish.github.io
14. *DirectFisheye-GS*: https://github.com/Metaverse-AI-Lab-THU/DirectFisheye-GS
15. *Fisheye-GS*, mã nguồn: https://github.com/zmliao/Fisheye-GS

**Mã nguồn**

- neurad-studio: https://github.com/georghess/neurad-studio
- SplatAD: https://github.com/carlinds/splatad
- gsplat: https://github.com/nerfstudio-project/gsplat

**Bộ dữ liệu**

- PandaSet: https://scale.com/open-av-datasets/pandaset
- KITTI-360: https://www.cvlibs.net/datasets/kitti-360/
- Waymo Open Dataset: https://waymo.com/open/
- nuScenes: https://www.nuscenes.org/

---

## 16. Phụ lục

### 16.1 Lệnh NeuRAD đã chạy trên Windows

```powershell
cd E:\AIthucchien\neurad-studio-main
& ".\.venv\Scripts\python.exe" "nerfstudio\scripts\train.py" neurad `
  --pipeline.datamanager.num_processes=0 `
  --pipeline.datamanager.train_num_rays_per_batch=1024 `
  --pipeline.datamanager.eval_num_rays_per_batch=1024 `
  pandaset-data `
  --data ".\data\pandaset" `
  --sequence 001 `
  --dataset_end_fraction=0.2 `
  --cameras front
```

### 16.2 Kiểm tra checkpoint

```python
import glob
print(glob.glob('/content/neurad-studio/outputs/**/*.ckpt', recursive=True))
```

### 16.3 Lưu môi trường đã chạy được

```bash
pip freeze > /content/drive/MyDrive/neurad_env_freeze.txt
```

### 16.4 Tài liệu liên quan

- Sơ đồ động: [`diagram.html`](diagram.html)
- Mindmap Mermaid: [`methods_datasets.mmd`](methods_datasets.mmd)
- Bản tóm tắt: [`research_summary.md`](research_summary.md)
