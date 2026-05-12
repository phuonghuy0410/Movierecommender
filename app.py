import streamlit as st
import pandas as pd
import numpy as np

from recommender import search_movies, search_movies_by_description, recommend_movies

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
)

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
    <style>
        .movie-card {
            background-color: #1e1e2e;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
            border: 1px solid #2e2e3e;
        }
        .movie-title {
            font-size: 20px;
            font-weight: 700;
            color: #e0e0ff;
            margin-bottom: 6px;
        }
        .movie-genre {
            font-size: 14px;
            color: #a0a0c0;
            margin-bottom: 10px;
        }
        .tag-pill {
            display: inline-block;
            background-color: #2e2e50;
            color: #b0b0ff;
            border-radius: 20px;
            padding: 2px 10px;
            font-size: 12px;
            margin: 2px 3px 2px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# LOAD DATA (CACHED)
# ==============================

@st.cache_data
def load_data():
    ratings = pd.read_csv("data/ratings.csv")
    tags = pd.read_csv("data/tags.csv")
    return ratings, tags

ratings, tags = load_data()

# ==============================
# HEADER
# ==============================

st.title("🎬 Movie Recommendation System")
st.markdown("Nhập **tên phim**, **từ khóa** hoặc **mô tả** bằng tiếng Việt hoặc tiếng Anh — hệ thống tự động gợi ý phim phù hợp.")
st.divider()

# ==============================
# HÀM RENDER KẾT QUẢ
# ==============================

def _render_results(result: pd.DataFrame):
    # Tính avg_rating cho từng phim rồi sort từ cao xuống thấp
    avg_ratings = (
        ratings.groupby("movieId")["rating"]
        .mean()
        .rename("avg_rating")
    )
    result = result.join(avg_ratings, on="movieId").sort_values(
        "avg_rating", ascending=False
    ).reset_index(drop=True)

    for i, (_, row) in enumerate(result.iterrows(), start=1):
        movie_id = row["movieId"]

        movie_ratings = ratings[ratings["movieId"] == movie_id].head(3)
        avg_rating = row.get("avg_rating")
        avg_rating = avg_rating if pd.notna(avg_rating) else None

        movie_tags = (
            tags[tags["movieId"] == movie_id]["tag"]
            .dropna()
            .head(5)
            .tolist()
        )

        genres_display = row.get("genres", "N/A").replace("|", " · ")

        with st.container():
            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">{i}. {row['title']}</div>
                <div class="movie-genre">🎭 {genres_display}</div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                if avg_rating is not None:
                    st.metric(
                        label="⭐ Đánh giá trung bình",
                        value=f"{avg_rating:.1f} / 5.0",
                    )
                if not movie_ratings.empty:
                    st.markdown("**Một số đánh giá người dùng:**")
                    for _, r in movie_ratings.iterrows():
                        stars = "⭐" * int(r["rating"])
                        st.markdown(
                            f"- User `{int(r['userId'])}`: {stars} `{r['rating']}`"
                        )

            

            st.divider()

# ==============================
# INPUT DUY NHẤT
# ==============================

query = st.text_area(
    "🔍 Nhập tên phim hoặc mô tả",
    placeholder=(
        "Ví dụ: Inception\n"
        "hoặc: Phim kinh dị về ngôi nhà ma ám\n"
        "hoặc: A comedy about two friends on a road trip"
    ),
    height=100,
)

if st.button("🎯 Tìm phim", use_container_width=True) and query.strip():

    # BƯỚC 1: THỬ TÌM THEO TÊN TRƯỚC
    matched = search_movies(query.strip())

    if matched is not None and not matched.empty:
        # Lấy phim đầu tiên trong kết quả, gợi ý luôn không cần chọn
        selected_movie = matched.iloc[0]["title"]

        with st.spinner("Đang tìm phim tương tự..."):
            result = recommend_movies(selected_movie)

        if result is None or result.empty:
            # Phim không có trong tập train → fallback semantic
            with st.spinner("Đang tìm theo mô tả..."):
                result = search_movies_by_description(query)

            if result is not None and not result.empty:
                st.subheader("🎯 Gợi ý phim tương tự")
                st.caption(f"Tìm thấy {len(result)} phim")
                st.divider()
                _render_results(result)
            else:
                st.error("❌ Không tìm thấy phim phù hợp.")
        else:
            st.subheader(f"📌 Phim gợi ý dựa trên: *{selected_movie}*")
            st.caption(f"Tìm thấy {len(result)} phim tương tự")
            st.divider()
            _render_results(result)

    # BƯỚC 2: KHÔNG KHỚP TÊN → SEMANTIC SEARCH
    else:
        with st.spinner("Đang phân tích mô tả..."):
            result = search_movies_by_description(query)

        if result is None or result.empty:
            st.error("❌ Không tìm thấy phim phù hợp. Hãy thử mô tả khác.")
        else:
            st.subheader("🎯 Gợi ý phim dựa trên mô tả của bạn")
            st.caption(f"Tìm thấy {len(result)} phim")
            st.divider()
            _render_results(result)