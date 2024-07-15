import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient
import os
import base64
import io
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize the Roboflow client
API_KEY = os.getenv("ROBOWFLOW_API_KEY")
MODEL_ID = "fashion-wpxvj/3"
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = "sleeves-necklines"

# Create a Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Function to draw bounding boxes based on model predictions
def draw_bounding_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    for pred in predictions:
        x1, y1 = pred['x'] - pred['width'] / 2, pred['y'] - pred['height'] / 2
        x2, y2 = pred['x'] + pred['width'] / 2, pred['y'] + pred['height'] / 2
        label = pred['class']
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), label, fill="red")
    return image

# Function to save data to Supabase
def save_data_to_supabase(image, predictions, description):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        data = {
            "image": img_str,
            "predictions": predictions,
            "description": description
        }
        
        response = supabase.table(SUPABASE_TABLE).insert(data).execute()
        
        if response and not response.get('error'):
            st.write("Response from Supabase:", response)
            st.success("Data saved to Supabase successfully!")
        else:
            st.error("Failed to save data to Supabase:", response['error'])
        
        return response
    except Exception as e:
        st.error(f"Error saving data to Supabase: {e}")
        return None

# Function to retrieve data from Supabase
def get_data_from_supabase(query=None):
    try:
        if query:
            response = supabase.table(SUPABASE_TABLE).select("*").filter(query).execute()
        else:
            response = supabase.table(SUPABASE_TABLE).select("*").execute()
        return response.data
    except Exception as e:
        st.error(f"Error retrieving data from Supabase: {e}")
        return []

# Streamlit app configuration
st.set_page_config(page_title="Roboflow Inference", page_icon=":tada:", layout="wide")

# Create a sidebar
page = st.sidebar.radio("WELCOME", ["Home", "Upload", "Search"])

# Home page
if page == "Home":
    st.title("Roboflow Inference with Bounding Boxes")
    st.write("Use the sidebar to navigate to Upload or Search pages.")

# Upload Image page
elif page == "Upload":
    st.title("Upload an Image")

    # Image upload
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Perform inference
        result = CLIENT.infer("temp_image.jpg", model_id=MODEL_ID)
        predictions = result['predictions']
        
        # Draw bounding boxes
        image_with_bounding_boxes = draw_bounding_boxes(image.copy(), predictions)
        
        # Display the image with bounding boxes
        st.image(image_with_bounding_boxes, caption='Image with Bounding Boxes', use_column_width=True)

        # Show inference result details
        st.write("Inference Results:")
        st.json(predictions)

        # Description input for Supabase
        description = st.text_input("Enter a description for the image:", "")

        # Save data to Supabase
        if st.button("Save Results to Database"):
            if description:
                response = save_data_to_supabase(image_with_bounding_boxes, predictions, description)
            else:
                st.warning("Please provide a description to save the data.")

# Search page
elif page == "Search":
    st.title("Search Stored Data")

    search_query = st.text_input("Enter search query (e.g., 'class=neckline'):")
    
    if st.button("Search"):
        data = get_data_from_supabase(search_query)
        
        if data:
            for record in data:
                img_str = record['image']
                predictions = record['predictions']
                
                img_data = base64.b64decode(img_str)
                image = Image.open(io.BytesIO(img_data))
                
                st.image(image, caption='Stored Image', use_column_width=True)
                st.write("Stored Predictions:")
                st.json(predictions)
        else:
            st.write("No data found for the given query.")
