# Phân Tích Ý Kiến Phản Hồi Sinh Viên (Student Feedback Sentiment Analysis)

## Tổng Quan

Trong kỷ nguyên số, dữ liệu văn bản từ sinh viên (phản hồi về môn học, giảng viên, cơ sở vật chất) là nguồn tài nguyên vô giá giúp nhà trường cải thiện chất lượng đào tạo. Tuy nhiên, việc đọc và phân loại thủ công hàng ngàn ý kiến là điều không khả thi.

Bài thi này yêu cầu sinh viên vận dụng kiến thức về **Xử lý ngôn ngữ tự nhiên (NLP)** để xây dựng mô hình phân tích ý kiến phản hồi.

## Bài Toán

Phân loại các phản hồi của sinh viên về môi trường đại học dựa trên **cảm xúc (sentiment)**.

### Cấu trúc nhãn Sentiment

| Giá trị | Ý nghĩa           |
|---------|--------------------|
| 0       | Tiêu cực (Negative)|
| 1       | Trung tính (Neutral)|
| 2       | Tích cực (Positive) |

### Cấu trúc nhãn Topic (chỉ có trong tập train)

| Giá trị | Ý nghĩa                       |
|---------|--------------------------------|
| 0       | Giảng viên (Lecturer)          |
| 1       | Chương trình đào tạo (Training program) |
| 2       | Cơ sở vật chất (Facility)      |
| 3       | Khác (Others)                  |

## Đánh Giá

Kết quả nộp bài được đánh giá dựa trên chỉ số **Macro F1-score**.

## Dữ Liệu

### Files
- **train.csv** - Tập huấn luyện chứa: `id`, `sentence`, `topic`, `sentiment`
- **test.csv** - Tập kiểm tra chứa: `id`, `sentence`
- **sample_submission.csv** - Tệp mẫu định dạng nộp bài

### Columns
| Cột        | Mô tả                                              |
|------------|-----------------------------------------------------|
| `id`       | Mã định danh cho mỗi phản hồi                      |
| `sentence` | Nội dung văn bản phản hồi của sinh viên             |
| `sentiment`| Nhãn cảm xúc (0, 1, 2) - chỉ có trong tập train    |
| `topic`    | Nhãn chủ đề (0, 1, 2, 3) - chỉ có trong tập train  |

### Thống kê
- Tập train: **11,323 mẫu**
- Tập test: **4,854 mẫu**

## Acronyms (Tiền xử lý ký tự đặc biệt)

Dữ liệu văn bản trong cột `sentence` đã được xử lý bằng cách thay thế các biểu tượng cảm xúc và ký tự đặc biệt:

| Ký tự gốc | Được thay thế bởi     |
|------------|----------------------|
| `:)`       | `colonsmile`         |
| `:(`       | `colonsad`           |
| `@@`       | `colonsurprise`      |
| `<3`       | `colonlove`          |
| `:d`       | `colonsmilesmile`    |
| `:3`       | `coloncontemn`       |
| `:v`       | `colonbigsmile`      |
| `:_`       | `coloncc`            |
| `:p`       | `colonsmallsmile`    |
| `>>`       | `coloncolon`         |
| `:">`      | `colonlovelove`      |
| `^^`       | `colonhihi`          |
| `:`        | `doubledot`          |
| `:'(`      | `colonsadcolon`      |
| `:@`       | `colondoublesurprise`|
| `v.v`      | `vdotv`              |
| `…`        | `dotdotdot`          |
| `/`        | `fraction`           |
| `c#`       | `csharp`             |

## Định Dạng Nộp Bài

File `.csv` có 2 cột: `id` và `sentiment`.

```csv
id,sentiment
0,2
1,0
2,1
...
```
