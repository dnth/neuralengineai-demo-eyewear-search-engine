import streamlit as st
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import os
from PIL import Image

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def init_milvus_client():
    return MilvusClient("nbs/milvus_lite.db")

def search_eyewear(query_text, limit=9):
    client = init_milvus_client()
    model = load_model()
    
    collection_name = "eyewear"
    
    search_results = client.search(
        collection_name=collection_name,
        data=[model.encode(query_text).tolist()],
        limit=limit,
        output_fields=["caption", "image_path", "product_type", "prompt"]
    )
    
    return search_results[0]

st.title("🕶️ Eyewear Search Engine")
st.write("Search for eyewear by describing what you're looking for!")

with st.sidebar:
    st.header("Search Options")
    query = st.text_input("Enter your search query:", placeholder="e.g., polaroid sunglasses, round glasses, blue frames")
    num_results = st.slider("Number of results:", min_value=1, max_value=50, value=9)

if query:
    with st.spinner("Searching..."):
        results = search_eyewear(query, num_results)
    
    st.write(f"Found {len(results)} results for: **{query}**")
    
    cols = st.columns(3)
    
    for i, result in enumerate(results):
        col_idx = i % 3
        
        with cols[col_idx]:
            image_path = os.path.join("nbs", result['entity']['image_path'])
            
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, use_container_width=True)
            else:
                st.error(f"Image not found: {image_path}")
            
            st.write(f"**Product Type:** {result['entity']['product_type']}")
            
            with st.expander("View Details"):
                st.write(f"**Distance:** {result['distance']:.3f}")
                st.write(f"**Caption:** {result['entity']['caption']}")
                st.write(f"**Prompt:** {result['entity']['prompt']}")
            
            st.divider()