# fflab: CLI cho Fisheye Fusion Lab

`fflab` là một CLI Node.js để cài đặt, train, render và kiểm tra các backend dựng cảnh 3D bằng Gaussian. Nó chạy được trên máy có GPU (Linux, Windows), server headless, hoặc điều khiển một VM Google Colab từ xa. Phần tính toán vẫn là các script Python trong `../tools`; `fflab` lo phần cài đặt, gọi đúng tham số và vận hành.

Hiện có một backend: **`3dgut-gsplat`** (3DGUT trên gsplat 1.5.3, KITTI-360 với 2 camera pinhole, 2 fisheye và LiDAR). Backend và framework tăng tốc khai báo bằng file JSON, nên thêm phương pháp mới là thêm một file manifest (xem "Mở rộng").

## Cài đặt

Cần Node.js 18.3 trở lên và Python 3.9 trở lên. Không có thư viện npm nào phải tải.

```bash
cd fflab
npm pack
npm install -g ./fflab-0.1.0.tgz
fflab doctor
```

Hoặc chạy thẳng từ repo, không cần cài: `node fflab/bin/fflab.js <lệnh>`.

## Các lệnh

| Lệnh | Việc làm | Cần GPU |
| --- | --- | --- |
| `fflab list` | liệt kê backend và framework tăng tốc | không |
| `fflab doctor` | kiểm tra Python, PyTorch, CUDA, GPU, colab CLI | không |
| `fflab setup` | tạo venv (`~/.fflab/venv`) và cài backend + accelerator | không |
| `fflab train` | train trên máy này | có |
| `fflab render` | video fly-through từ file `.ply` | có |
| `fflab check` | kiểm tra chiếu LiDAR vào ảnh | không |
| `fflab colab train\|render\|check` | chạy đúng lệnh đó trên VM Colab | VM lo |
| `fflab colab logs` / `fflab colab stop` | theo dõi lại job / tắt VM | không |
| `fflab help train` | các tham số của một lệnh | không |

Mọi lệnh chạy đều có `--dry-run` để in kế hoạch (biến môi trường, script) mà không chạy gì.

## Ví dụ

Máy có GPU NVIDIA (Linux hoặc Windows):

```bash
fflab setup --accel fused-ssim
fflab train --frames 100 --steps 10000 --accel fused-ssim --run-name road1
fflab render --run-name road1
```

Windows: wheel PyTorch trên PyPI cho Windows chỉ chạy CPU, nên cần `fflab setup --torch-index-url https://download.pytorch.org/whl/cu124` (chọn theo driver CUDA). Compile gsplat trên Windows cần CUDA toolkit và Visual Studio Build Tools (C++).

Server không có GPU, hoặc chỉ muốn dùng Colab (Linux, macOS, hoặc WSL trên Windows):

```bash
uv tool install google-colab-cli --with keyrings.alt
export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring   # máy headless không có keyring
fflab colab train --gpu T4 --run-name road1 --sync-dir /content/drive/MyDrive/fflab --drive
fflab colab render --run-name road1
fflab colab stop
```

`fflab colab` sẽ:

1. tạo hoặc dùng lại phiên Colab;
2. mount Drive nếu có `--drive` (in link để bạn cấp quyền rồi bấm Enter);
3. tải script lên VM;
4. chỉ cài gói nào còn thiếu;
5. chạy job ở nền trên VM kèm file log;
6. đọc log theo từng đoạn, thử lại khi rớt kết nối;
7. tải kết quả về `./fflab-runs/<run>` (không kèm checkpoint và PLY giữa chừng, trừ khi có `--with-ckpt`).

Ctrl+C chỉ dừng việc theo dõi, job vẫn chạy trên VM; dùng `fflab colab logs` để theo dõi lại.

## Tham số train chính (`3dgut-gsplat`)

| Cờ | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--data` | `./kitti360` | thư mục archive; thiếu file nhỏ thì tự tải, thiếu tar thì stream từ Hugging Face |
| `--frames` / `--steps` | 100 / 10000 | số frame của sequence 0003, số bước train |
| `--cap` | 1000000 | trần số Gaussian (MCMC) |
| `--sky` / `--ssim` / `--mask` | bật | bầu trời (tác dụng lớn nhất), SSIM, bỏ qua thân xe |
| `--sync-dir` | trống | chép kết quả và checkpoint sang thư mục này (vd. Drive) |
| `--resume` | tắt | chạy tiếp run cùng tên từ checkpoint cuối |

Danh sách đầy đủ: `fflab help train`.

## Framework tăng tốc

| ID | Trạng thái | Ghi chú |
| --- | --- | --- |
| `selective-adam` | có sẵn | chỉ cập nhật Gaussian đang nhìn thấy (kiểu Fast3DGS) |
| `mcmc` | có sẵn | tăng số Gaussian không cần gradient 2D (bắt buộc với 3DGUT) |
| `fused-ssim` | thử nghiệm | kernel CUDA thay cho SSIM bằng conv2d. Đo một lần trên T4: train 215 s thay vì 487 s, PSNR giữ nguyên |

Bật bằng `--accel id1,id2`. Accelerator có sẵn luôn bật; ghi tên nó chỉ in ra một dòng ghi chú.

## Mở rộng: thêm backend hoặc accelerator

Đặt file JSON vào `backends/` hoặc `accelerators/` (hoặc vào `$FFLAB_PLUGINS_DIR/backends|accelerators` để không phải sửa gói).

- **Backend** khai báo: các script Python cho từng lệnh, gói cần cài (`requirements`, `buildRequirements` cài với `--no-build-isolation`), module để kiểm tra đã cài chưa (`importCheck`), và các tham số. Mỗi tham số gồm `flag`, `env`, `type` (`path`, `string`, `int`, `float`, `bool`), `default`, `help`. Script đọc tham số qua biến môi trường.
- **Accelerator** khai báo: các backend nó hỗ trợ, gói cần cài và biến môi trường nó bật. Ví dụ `fused-ssim` đặt `KITTI360_SSIM_IMPL=fused`.

`fflab` luôn đặt rõ **mọi** tham số của lệnh, kể cả giá trị mặc định. Nhờ vậy một biến môi trường sót lại trong shell hay trong kernel Colab không thể âm thầm làm đổi kết quả (lỗi này đã thực sự xảy ra khi chạy so sánh).

## Kiểm thử

```bash
npm test
```

Test gồm phần đơn vị, một lần chạy `fflab check` thật trên archive KITTI-360 giả lập (chỉ cần CPU), và toàn bộ quy trình `fflab colab` trên một bản `colab` giả (`test/fixtures/fake-colab`).

## Giới hạn hiện tại

- `fflab train` và `fflab render` chưa chạy thử trên máy GPU cục bộ; chúng gọi đúng các script đã chạy trên Colab T4.
- `fused-ssim` mới đo một lần (một VM T4); chưa đo nhiễu giữa các lần chạy.
- Lần đầu trên mỗi VM, gsplat compile kernel CUDA khoảng 13–15 phút (793 s trong lần đo); it/s in ra trong lúc đó bị kéo thấp.
- `google-colab-cli` không hỗ trợ Windows, nên `fflab colab` trên Windows phải chạy qua WSL.
- Backend hiện chỉ đọc sequence 0003 của KITTI-360 (giới hạn của trainer).
