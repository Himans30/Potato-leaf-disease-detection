import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
MODEL_PATH = "potatoes.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels manually (Update as per your model)
CLASS_LABELS = ["Healthy", "Early Blight", "Late Blight"]

# Disease details
DISEASE_DETAILS = {
    "Healthy": {
        "description": "Your plant is healthy! No disease detected.",
        "solution": "Keep providing optimal sunlight, water, and nutrients."
    },
    "Early Blight": {
        "description": "Early Blight is a fungal disease causing dark spots on leaves.",
        "solution": "Use fungicides and remove infected leaves."
    },
    "Late Blight": {
        "description": "Late Blight spreads rapidly and affects leaves and stems.",
        "solution": "Use copper-based fungicides and maintain proper ventilation."
    }
}

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB
    image = image.resize((128, 128))  # Resize to match model input size
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI Setup
st.set_page_config(page_title="Potato Disease Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Welcome", ["Home", "About", "Prediction"])

# Light Background Theme

# Home Page
if page == "Home":
    st.title("🥔 Welcome to the Potato Disease Prediction System")
    st.write(
        """
        This system helps in detecting diseases in potato leaves using Deep Learning.
        **Why Use This?**
        - Prevent crop loss by identifying diseases early.
        - Easy and quick disease detection with a simple image upload.
        - Provides treatment suggestions to keep your plants healthy.
        """
    )
    
    # Image
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/85/Potato_plants.jpg", use_container_width=True)
    
    # How It Works Section
    st.subheader("🚀 How It Works?")
    st.write("""
    1. **Upload a photo** of a potato leaf.
    2. The AI model **analyzes the image** and identifies any diseases.
    3. It then **displays the disease name** along with **treatment suggestions**.
    4. Take **corrective action** to keep your crop healthy!  
    """)

    # Common Potato Diseases
    st.subheader("🌿 Common Potato Diseases")
    disease_info = """
    - **Healthy:** No issues detected, maintain optimal conditions.
    - **Early Blight:** Causes dark spots; treated with fungicides.
    - **Late Blight:** Highly contagious; requires copper-based fungicides.
    """
    st.info(disease_info)

# About Page
elif page == "About":
    st.title("ℹ️ About This Project")
    st.write(
        """
        This project is built using **Streamlit** and **TensorFlow** to predict potato diseases.
        It is designed to help farmers and researchers detect plant diseases early.
        """
    )

    # Dataset Info
    st.subheader("📊 Dataset Details")
    st.write("""
    - **Data Source:** Publicly available Potato Disease Dataset
    - **Classes:** Healthy, Early Blight, Late Blight
    - **Total Images:** 5000+ classified images used for training
    - **Model:** Trained using Deep Learning (CNN-based architecture)
    """)

    # Technology Used
    st.subheader("🛠️ Technology Used")
    st.write("""
    - **Frontend:** Streamlit (Python)
    - **Backend:** TensorFlow for AI model
    - **Libraries:** OpenCV, NumPy, Pillow
    """)

    # Use Cases
    st.subheader("🌍 Use Cases")
    use_cases = """
    - **Farmers:** Helps in early detection to prevent crop loss.
    - **Researchers:** Useful for agricultural disease analysis.
    - **Agriculture Experts:** Assists in large-scale disease monitoring.
    """
    st.success(use_cases)

# Prediction Page
elif page == "Prediction":
    st.title("🔎 Potato Disease Prediction")
    st.write("Upload an image of a potato leaf to predict the disease.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)  # Get the highest probability index
        confidence = np.max(prediction) * 100  # Convert to percentage

        # Get the disease label
        disease_name = CLASS_LABELS[predicted_class] if predicted_class < len(CLASS_LABELS) else "Unknown"

        # Get disease details
        disease_info = DISEASE_DETAILS.get(disease_name, {"description": "Unknown", "solution": "No solution available."})

        # Display prediction results
        st.subheader("🧪 Prediction Results:")
        st.write(f"**Predicted Disease:** {disease_name}")
        st.write(f"**Confidence Level:** {confidence:.2f}%")
        st.write(f"**Description:** {disease_info['description']}")
        st.write(f"**Solution:** {disease_info['solution']}")

        # Show success message
        st.success("✅ Prediction Complete!")

        # Option to analyze another image
        if st.button("🔄 Analyze Another Image"):
            st.rerun()

