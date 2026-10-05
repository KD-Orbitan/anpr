# Chọn bản công khai — không cần Codex

Bạn luôn làm việc trong một repo `anpr_project`. Hai bản được **xuất từ cùng code**, không duy trì hai nhánh code khác nhau dễ bị lệch. Các gói xuất nằm trong `.private/releases/`, Git bỏ qua toàn bộ thư mục này.

| Bản | Có gì? | Không có gì? |
|---|---|---|
| `code-only` | Code, tài liệu, tests, dictionary và metadata model | Trọng số, số liệu AT, ảnh/nhãn, dự đoán từng ảnh |
| `full` | Cùng code + trang kết quả tổng hợp + ZIP trọng số OCR | Ảnh/nhãn AT, tên công ty, tên ảnh, dự đoán từng ảnh, detector YOLO, pretrained gốc |

“Full” chỉ có nghĩa đủ ba quyền đã bàn: code, **OCR fine-tuned weights**, và số liệu tổng hợp. Không tự mở rộng quyền sang detector, dữ liệu hay pretrained bên thứ ba. Thông tin lựa chọn checkpoint/tập đánh giá vẫn phải được mô tả trung thực.

## 1. Chuẩn bị/xem hai bản ngay khi chưa có quyền

Mở terminal tại `anpr_project`, dùng Python 3.9 trở lên. Các lệnh xuất bản này không cần cài Paddle hay môi trường ML:

```powershell
python scripts/release_profiles.py build code-only
python scripts/release_profiles.py build full
```

Mỗi lệnh in đường dẫn một gói mới, không ghi đè gói cũ, không commit và không push. Mở `source/README.md` trong gói để xem.

Điểm mở nhanh cố định trên máy: **`.private/releases/README.md`**, có liên kết tới hai bản mới nhất.

- `source/`: phần sẽ đưa vào Git.
- `assets/`: chỉ có ở bản full, gồm `ocr-models.zip` và `evaluation-summary.json`; để đính kèm GitHub Release, không đưa vào Git history.
- `bundle.json`: danh sách file và checksum để phát hiện thay đổi sau khi xuất; không phải giấy xác nhận quyền.
- `.private/releases/latest-code-only.json` và `latest-full.json`: chỉ tới bản xuất mới nhất.

Bản full cần các trọng số local và `.private/release-inputs.json` đã được thiết lập trên máy này. File input chỉ chọn các báo cáo local cần tổng hợp. Nó **không được push**. Script chỉ lấy các trường số đã định nghĩa, không sao chép nguyên JSON báo cáo chứa đường dẫn riêng. Người clone repo từ GitHub không tự có những input private đó.

## 2. Sau khi được xác nhận quyền: chọn một bản

Chỉ code được phép:

```powershell
python scripts/release_profiles.py stage code-only --confirm-permissions
```

Cả code, trọng số OCR và số liệu tổng hợp được phép:

```powershell
python scripts/release_profiles.py stage full --confirm-permissions
```

`--confirm-permissions` là xác nhận **do bạn đưa ra**, không phải script tự kiểm tra hợp đồng. Khi chưa rõ quyền, chỉ dùng `build` để xem trước.

Lệnh `stage` kiểm tra checksum, tạo một nhánh `publish/...` và một Git worktree riêng trong `.private/publish/`, rồi commit bản đã chọn. Nó không sửa nhánh main hoặc dữ liệu trong folder bạn đang làm việc, không push và không tự công bố assets. Bắt đầu từ HEAD hiện tại của repo; trước khi stage nên cập nhật nhánh làm việc về phiên bản bạn muốn phát hành.

## 3. Đưa lên GitHub bằng các thao tác thông thường

1. Xem nội dung worktree mà lệnh vừa in, đặc biệt README và trang kết quả ở bản full.
2. Chạy **đúng lệnh `git ... push` mà script in ra**.
3. Mở repo GitHub, chọn “Compare & pull request” cho nhánh `publish/...`; kiểm tra Files changed và CI rồi merge vào `main`.
4. Với full: vào **Releases → Draft a new release**, tạo tag mới, đính kèm đúng hai file trong `assets/` của gói vừa chọn. Chỉ bấm Publish release khi đã được phép.
5. Với code-only: không tải assets lên. Không cần GitHub Release để public code.

Tài liệu hướng dẫn luôn nằm trong repo này nên bạn không cần quay lại hội thoại hoặc dùng Codex. Không gửi cả folder `.private` hay ZIP toàn bộ thư mục làm việc.

## Chuyển lựa chọn sau này

Trước khi công khai, có thể build/stage lại bản khác tùy ý. Nếu đã public bản full, chuyển về code-only chỉ thay bản hiện tại; nó không xóa kết quả khỏi Git history, các Release attachments, hoặc bản người khác đã tải. Muốn thu hồi cần xử lý lịch sử/attachments riêng, không dùng việc đổi profile như cơ chế thu hồi.

Bản full lưu template tài liệu code-only trong `configs/code-only-docs.json`, nên khi xuất lại code-only, trang kết quả sẽ được loại và README được khôi phục. Nếu sửa README trong bản full, hãy cập nhật template công khai này khi muốn thay đổi cả README code-only.

Hai bản không tự cấp license. Chỉ chọn/đổi license sau khi xác nhận quyền với code và thành phần bên thứ ba. Repo hiện vẫn giữ nguyên các giới hạn quyền đã ghi trong notices.
