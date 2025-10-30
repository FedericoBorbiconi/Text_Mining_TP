import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ======================================================
# 1. Load model and FAISS index with caching
# ======================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

@st.cache_resource
def load_index():
    return faiss.read_index("libros_clean.index")

@st.cache_data
def load_metadata():
    return pd.read_parquet("libros_clean_metadata.parquet", engine="fastparquet")


# ======================================================
# 2. Helper functions
# ======================================================
def build_expanded_query(query: str) -> str:
    """Ask for more details if the query is too short."""
    if len(query.strip()) < 40:
        st.warning("Your query seems too short. Please answer a few questions so we can refine it.")
        
        genre = st.text_input("📚 Genre (e.g., romance, fantasy, mystery):")
        place = st.text_input("🌍 Setting (e.g., small town, big city, magical kingdom, outer space):")
        characters = st.text_input("👤 Main characters (e.g., teenagers, heroes, families):")
        tone = st.text_input("🎭 Tone (e.g., dramatic, funny, dark, hopeful):")
        
        if genre or place or characters or tone:
            query_expanded = f"A {genre} story set in {place}, featuring {characters} characters, with a {tone} tone."
            st.info(f"Expanded query: {query_expanded}")
            return query_expanded
    else:
        st.write(f"🗣️ Query: {query}")

    return query.strip()


def filter_results(df, filters):
    """Filter results based on author or last book."""
    results = df
    if "author" in filters and filters["author"]:
        results = results[results["authors"].str.lower().str.contains(filters["author"].lower())]
    if "last_book" in filters and filters["last_book"]:
        mask = results["title"].str.lower().str.contains(filters["last_book"].lower())
        results = results[~mask]
    return results


# ======================================================
# 3. Main app
# ======================================================
def main():
    st.title("📖 Fiction Book Recommender")
    st.write("Find fiction books that match your preferences based on descriptions and ratings.")

    model = load_model()
    index = load_index()
    df = load_metadata()

    # --- Handle refined query from previous step ---
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "pending_refine" in st.session_state and st.session_state.pending_refine:
        # user just clicked "Search again"
        query_input = st.session_state.query_input
        st.session_state.pending_refine = False  # consume flag
    else:
        query_input = st.text_input(
            "🔎 What kind of book are you looking for?",
            value=st.session_state.get("query_input", ""),
            key="query_input_box"
        )
        st.session_state.query_input = query_input

    if not query_input.strip():
        st.stop()

    # --- Expand query if needed ---
    query_final = build_expanded_query(query_input)
    if not query_final:
        st.warning("Please provide more details to proceed.")
        st.stop()

    st.session_state.query_final = query_final

    # --- Encode and search ---
    query_embedding = model.encode([query_final], normalize_embeddings=True)
    k = 100
    distances, indices = index.search(query_embedding.astype("float32"), k)
    results_df = df.iloc[indices[0]].copy()
    results_df["similarity"] = distances[0]

    # --- Filters ---
    with st.expander("⚙️ Add filters"):
        author = st.text_input("Filter by author (optional):")
        last_book = st.text_input("Exclude last book you read (optional):")
        filters = {"author": author, "last_book": last_book}
    filtered_df = filter_results(results_df, filters)

    # --- Rating consideration ---
    consider_rating = st.checkbox("⭐ Consider average rating in ranking", value=True)
    if consider_rating:
        if filtered_df["avg_rating"].notna().any():
            min_r, max_r = filtered_df["avg_rating"].min(), filtered_df["avg_rating"].max()
            if max_r > min_r:
                filtered_df["rating_norm"] = (filtered_df["avg_rating"] - min_r) / (max_r - min_r)
            else:
                filtered_df["rating_norm"] = 0.5
        else:
            filtered_df["rating_norm"] = 0.2

        filtered_df["similarity"] = (
            0.9 * filtered_df["similarity"] + 0.1 * filtered_df["rating_norm"]
        )
        filtered_df = filtered_df.sort_values(by="similarity", ascending=False)

    if filtered_df.empty:
        st.error("😔 Sorry, no books matched your request.")
        st.stop()

    # --- Display results ---
    if "show_more" not in st.session_state:
        st.session_state.show_more = False

    if not st.session_state.show_more:
        display_df = filtered_df.head(5)
        st.success("✅ Showing top 5 recommendations:")
        start_rank = 1
    else:
        display_df = filtered_df.iloc[5:10]
        st.info("📚 Showing additional recommendations (ranks 6–10):")
        start_rank = 6

    for i, (_, row) in enumerate(display_df.iterrows(), start=start_rank):
        with st.container():
            st.subheader(f"{i}. {row['title']}")
            st.write(f"**Author:** {row['authors']}")
            st.write(f"**Similarity score:** {row['similarity']:.3f}")
            st.write(f"**Average rating:** {row.get('avg_rating', 'N/A')}")
            st.write(row['description'][:400] + "...")

    # --- Feedback section ---
    st.markdown("---")
    st.write("Did you like these recommendations?")
    col1, col2 = st.columns(2)

    if "feedback_mode" not in st.session_state:
        st.session_state.feedback_mode = False
    if "refine_mode" not in st.session_state:
        st.session_state.refine_mode = False

    with col1:
        if st.button("👍 Yes"):
            st.success("🎉 Great! We're glad you liked the suggestions!")
            st.session_state.feedback_mode = False
            st.session_state.show_more = False
            st.session_state.refine_mode = False

    with col2:
        if st.button("👎 No"):
            st.session_state.feedback_mode = True
            st.session_state.show_more = False
            st.session_state.refine_mode = False
            st.rerun()

    if st.session_state.feedback_mode:
        st.warning("😔 Sorry to hear that.")
        action = st.radio(
            "What would you like to do next?",
            ("See more results (ranks 6–10)", "Refine my query"),
            key="feedback_action"
        )

        if st.button("Confirm"):
            if action == "See more results (ranks 6–10)":
                st.session_state.show_more = True
                st.session_state.feedback_mode = False
                st.session_state.refine_mode = False
                st.rerun()
            elif action == "Refine my query":
                st.session_state.feedback_mode = False
                st.session_state.show_more = False
                st.session_state.refine_mode = True
                st.session_state.last_query_input = st.session_state.query_input
                st.session_state.last_query_final = st.session_state.query_final
                st.rerun()

    if st.session_state.refine_mode:
        st.markdown("### ✏️ Refine your search")
        prefill = st.session_state.get("last_query_final") or st.session_state.get("last_query_input") or ""
        new_query = st.text_input(
            "🔍 What kind of book are you looking for?",
            value=prefill,
            key="refine_input"
        )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Search again"):
                st.session_state.query_input = new_query
                st.session_state.query_final = new_query
                st.session_state.refine_mode = False
                st.session_state.feedback_mode = False
                st.session_state.show_more = False
                st.session_state.pending_refine = True  # mark for next rerun
                st.rerun()
        with col_b:
            if st.button("Cancel"):
                st.session_state.refine_mode = False
                st.session_state.feedback_mode = False
                st.rerun()

        st.stop()


if __name__ == "__main__":
    main()
