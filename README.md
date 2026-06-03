# 🎬 Movie Recommender System

## 📌 Giới thiệu

Đây là dự án xây dựng hệ thống gợi ý phim sử dụng Python và Machine Learning.

Mục tiêu của hệ thống là:

* Gợi ý các bộ phim phù hợp với sở thích người dùng.
* Tìm các bộ phim tương tự dựa trên nội dung.
* Hỗ trợ nhập từ khóa hoặc mô tả để đề xuất phim liên quan.

Dự án phù hợp cho:

* Đồ án môn học
* Nghiên cứu Recommendation System
* Học Machine Learning / Data Science
* Thực hành xử lý dữ liệu với Python
---
# ⚙️ Công nghệ sử dụng

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* TF-IDF Vectorizer
* Cosine Similarity
---
# 🧠 Ý tưởng hệ thống

Hệ thống hoạt động theo hướng Collaborative Filtering.

Quy trình:

1. Thu thập dữ liệu phim.
2. Tiền xử lý dữ liệu.
3. Chuyển đổi nội dung phim thành vector bằng TF-IDF.
4. Tính độ tương đồng bằng Cosine Similarity.
5. Trả về danh sách phim tương tự.

---

# 📊 Thuật toán sử dụng

## Collaborative Filtering

Hệ thống đề xuất phim dựa trên:

* Thể loại
* Từ khóa
* Mô tả nội dung
* Metadata của phim

## TF-IDF Vectorizer

Dùng để chuyển văn bản thành vector số.

TF-IDF giúp:

* Giảm ảnh hưởng của từ xuất hiện quá nhiều
* Tăng trọng số cho từ quan trọng

## Cosine Similarity

Dùng để đo độ giống nhau giữa hai vector phim.

Giá trị càng gần 1 → phim càng giống nhau.

---

# 🚀 Cách chạy project

## 1. Clone project

```bash
git clone <repository_url>
cd movie_recommender
```

## 2. Cài thư viện

```bash
pip install -r requirements.txt
```

---

## 3. Chạy notebook
Mở file:

```bash
train.ipynb
```
## 4. Chạy Demo

```bash
streamlit run app.py
```

---



# 📈 Hướng phát triển

Có thể mở rộng thêm:

* Hybrid Recommendation System
* Deep Learning Recommendation
* Giao diện Web bằng Flask/Django
* API Recommendation

---

# 📚 Kiến thức áp dụng

* Machine Learning
* Natural Language Processing (NLP)
* Recommendation System
* Data Preprocessing
* Vector Space Model
* Similarity Measurement

---

# 👨‍💻 Tác giả

Lê Phương Huy - Hoàng Bảo Châu

---

# 📄 License

Dự án phục vụ mục đích học tập và nghiên cứu.
