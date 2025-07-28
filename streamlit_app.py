import streamlit as st
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import os
from PIL import Image
import timm
import torch

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_image_model():
    model = timm.create_model(
        'naflexvit_base_patch16_gap.e300_s576_in1k',
        pretrained=True,
        num_classes=0,  
    )
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    return model, transforms, device

@st.cache_resource
def init_milvus_client():
    return MilvusClient("nbs/milvus_lite.db")

def check_collections():
    """Check if required collections exist"""
    try:
        client = init_milvus_client()
        text_collection = "eyewear"
        image_collection = "eyewear_images_embeddings"
        
        has_text = client.has_collection(text_collection)
        has_image = client.has_collection(image_collection)
        
        if not has_text:
            st.warning(f"Text search collection '{text_collection}' not found.")
        if not has_image:
            st.warning(f"Image search collection '{image_collection}' not found.")
            
        return has_text and has_image
    except Exception as e:
        st.error(f"Error connecting to Milvus: {e}")
        return False

def search_eyewear(query_text, limit=9):
    try:
        client = init_milvus_client()
        model = load_model()
        
        collection_name = "eyewear"
        
        # Check if collection exists
        if not client.has_collection(collection_name):
            st.error(f"Collection '{collection_name}' not found. Please run the indexing notebook first.")
            return []
        
        # Encode the text query
        query_embedding = model.encode(query_text).tolist()
        
        search_results = client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=["caption", "image_path", "product_type", "prompt"]
        )
        
        return search_results[0]
    except Exception as e:
        st.error(f"Error during text search: {e}")
        return []

def search_eyewear_by_image(uploaded_image, limit=9, exclude_self=False):
    try:
        client = init_milvus_client()
        model, transforms, device = load_image_model()
        
        collection_name = "eyewear_images_embeddings"
        
        # Check if collection exists
        if not client.has_collection(collection_name):
            st.error(f"Collection '{collection_name}' not found. Please run the indexing notebook first.")
            return []
        
        # Preprocess and embed the uploaded image
        image_tensor = transforms(uploaded_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_embedding = model(image_tensor)
        
        # Convert to list format for Milvus
        image_embedding_list = image_embedding.cpu().tolist()[0]
        
        # If excluding self, search for more results and filter out perfect matches
        search_limit = limit + 1 if exclude_self else limit
        
        search_results = client.search(
            collection_name=collection_name,
            data=[image_embedding_list],
            limit=search_limit,
            output_fields=["caption", "image_path", "product_type", "prompt"]
        )
        
        results = search_results[0]
        
        # Filter out self-matches if requested
        if exclude_self:
            filtered_results = []
            for result in results:
                # Check if this is likely a self-match (distance very close to 0)
                if result['distance'] > 0.001:  # Small threshold to account for floating point precision
                    filtered_results.append(result)
                if len(filtered_results) >= limit:
                    break
            return filtered_results
        
        return results
    except Exception as e:
        st.error(f"Error during image search: {e}")
        return []

st.title("🕶️ Eyewear Search Engine")
st.write("Search for eyewear by describing what you're looking for!")

# Check if collections are available
if not check_collections():
    st.error("⚠️ Database not ready! Please run the indexing notebooks first:")
    st.markdown("""
    1. Run `04_embed-image.ipynb` to generate image embeddings
    2. Run `03_milvus-index-image.ipynb` to create the search index
    """)
    st.stop()
else:
    st.success("✅ Database ready for search!")

with st.sidebar:
    st.header("Search Options")
    
    search_mode = st.radio("Search by:", ["Text", "Image"])
    
    if search_mode == "Text":
        query = st.text_input("Enter your search query:", placeholder="e.g., polaroid sunglasses, round glasses, blue frames")
        uploaded_image = None
    else:
        uploaded_image = st.file_uploader("Upload an eyewear image:", type=['png', 'jpg', 'jpeg'])
        query = None
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    num_results = st.slider("Number of results:", min_value=1, max_value=50, value=9)

if query:
    with st.spinner("Searching..."):
        results = search_eyewear(query, num_results)
    
    st.write(f"Found {len(results)} results for: **{query}**")
elif uploaded_image:
    with st.spinner("Searching..."):
        image = Image.open(uploaded_image)
        results = search_eyewear_by_image(image, num_results, exclude_self=True)
    
    st.write(f"Found {len(results)} results for uploaded image")

if query or uploaded_image:
    if not results:
        st.warning("No results found. Please try a different search query or image.")
    else:
        cols = st.columns(3)
        
        for i, result in enumerate(results):
            col_idx = i % 3
            
            with cols[col_idx]:
                image_path = os.path.join("nbs", result['entity']['image_path'])
                
                if os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        st.image(image, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                else:
                    st.error(f"Image not found: {image_path}")
                
                st.write(f"**Product Type:** {result['entity']['product_type']}")
                
                with st.expander("View Details"):
                    st.write(f"**Distance:** {result['distance']:.3f}")
                    st.write(f"**Caption:** {result['entity']['caption']}")
                    st.write(f"**Prompt:** {result['entity']['prompt']}")
                
                st.divider()