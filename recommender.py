import pandas as pd
import numpy as np
import pickle
import os
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# ==============================
# LOAD DỮ LIỆU
# ==============================

movies = pd.read_csv("data/movies_clean.csv")
tags = pd.read_csv("data/tags.csv")

# Load ma trận embedding phim — shape: (số_phim, 50)
# Mỗi hàng là vector 50 chiều đại diện cho "đặc trưng ẩn" của 1 phim
movie_embeddings = np.load("movie_embeddings.npy")

# Load các dictionary ánh xạ movieId ↔ index
with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

# movie2encoded: movieId (gốc) → index trong ma trận embedding
movie2encoded = mappings["movie2encoded"]

# encoded2movie: index → movieId (gốc)
encoded2movie = mappings["encoded2movie"]

# Tạo set các movieId có ít nhất 1 tag từ người dùng
movies_with_tags = set(tags["movieId"].unique())

# ==============================
# SEMANTIC SEARCH SETUP
# ==============================

# Model đa ngôn ngữ — hỗ trợ tiếng Việt và tiếng Anh
# Chạy 1 lần lúc khởi động, Streamlit cache lại sau đó
semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

movies["_text"] = movies["title"] + " " + movies["genres"].str.replace("|", " ", regex=False)

# Nếu đã có file cache → load luôn, không encode lại
if os.path.exists("movie_text_embeddings.npy"):
    movie_text_embeddings = np.load("movie_text_embeddings.npy")
else:
    movie_text_embeddings = semantic_model.encode(
        movies["_text"].tolist(),
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=64,
    )
    np.save("movie_text_embeddings.npy", movie_text_embeddings)

# Tạo text đại diện cho mỗi phim: tên + thể loại
movies["_text"] = movies["title"] + " " + movies["genres"].str.replace("|", " ", regex=False)

# Encode toàn bộ phim thành vector semantic (chạy 1 lần)
movie_text_embeddings = semantic_model.encode(
    movies["_text"].tolist(),
    show_progress_bar=False,
    convert_to_numpy=True,
    batch_size=64,
)

# ==============================
# HÀM TÌM KIẾM THEO TÊN (GIỮ LẠI)
# ==============================

def search_movies(keyword: str) -> pd.DataFrame:
    """
    Tìm phim theo từ khóa tên — hỗ trợ fuzzy nếu không khớp chính xác.
    """
    exact = movies[movies["title"].str.contains(keyword, case=False, na=False)]
    if not exact.empty:
        return exact

    titles = movies["title"].tolist()
    matches = process.extract(
        keyword,
        titles,
        scorer=fuzz.partial_ratio,
        limit=10,
        score_cutoff=70,
    )

    if not matches:
        return pd.DataFrame()

    matched_titles = [m[0] for m in matches]
    return movies[movies["title"].isin(matched_titles)]


# ==============================
# HÀM TÌM KIẾM THEO MÔ TẢ (MỚI)
# ==============================

def search_movies_by_description(query: str, top_n: int = 10) -> pd.DataFrame:
    """
    Tìm phim theo câu mô tả tự nhiên — hỗ trợ tiếng Việt và tiếng Anh.

    Cơ chế:
      1. Dịch query sang tiếng Anh (nếu là tiếng Việt)
      2. Encode query thành vector semantic
      3. Tính cosine similarity với toàn bộ phim
      4. Trả về top_n phim khớp nhất
    """
    # Bước 1: Dịch sang tiếng Anh
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(query)
    except Exception:
        translated = query  # fallback nếu mạng lỗi hoặc dịch thất bại

    # Bước 2: Encode câu query
    query_embedding = semantic_model.encode([translated], convert_to_numpy=True)

    # Bước 3: Tính cosine similarity với toàn bộ phim
    scores = cosine_similarity(query_embedding, movie_text_embeddings)[0]

    # Bước 4: Lấy top_n phim có score cao nhất
    top_indices = scores.argsort()[-top_n:][::-1]

    return movies.iloc[top_indices][["movieId", "title", "genres"]].copy()


# ==============================
# HÀM GỢI Ý PHIM TƯƠNG TỰ
# ==============================

def recommend_movies(movie_title: str, top_n: int = 10) -> pd.DataFrame | None:
    """
    Gợi ý phim tương tự dựa trên embedding đã train.

    Cơ chế:
      1. Tìm vector embedding của phim được chọn
      2. Tính cosine similarity với TẤT CẢ phim còn lại
      3. Lấy top phim có similarity cao nhất
      4. Ưu tiên phim có tag; fallback nếu không đủ
    """
    selected = movies[movies["title"] == movie_title]
    if selected.empty:
        return None

    movie_id = selected.iloc[0]["movieId"]

    # Phim không có trong tập train (bị loại ở bước remove noise)
    if movie_id not in movie2encoded:
        return None

    movie_index = movie2encoded[movie_id]
    target_embedding = movie_embeddings[movie_index].reshape(1, -1)

    # Tính cosine similarity với toàn bộ ma trận embedding
    similarity = cosine_similarity(target_embedding, movie_embeddings)[0]

    top_indices = similarity.argsort()[-(top_n * 3 + 1):][::-1]

    # Ưu tiên phim có tag
    recommended_ids = [
        encoded2movie[idx]
        for idx in top_indices
        if encoded2movie[idx] != movie_id
        and encoded2movie[idx] in movies_with_tags
    ][:top_n]

    # Fallback: bỏ điều kiện tag nếu không đủ kết quả
    if len(recommended_ids) < top_n:
        recommended_ids = [
            encoded2movie[idx]
            for idx in top_indices
            if encoded2movie[idx] != movie_id
        ][:top_n]

    if not recommended_ids:
        return None

    return movies[movies["movieId"].isin(recommended_ids)][
        ["movieId", "title", "genres"]
    ]