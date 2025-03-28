# Potato Disease Prediction System

## Overview

The **Potato Disease Prediction System** is a web-based application developed using **Streamlit**. It utilizes a trained deep learning model (`.h5` file) to classify potato diseases based on input images. The UI includes a sidebar with navigation options for **Home, About, and Prediction** pages.

## Features

- **Home Page:** Provides an introduction to the application.
- **About Page:** Explains the purpose and working of the application.
- **Prediction Page:** Allows users to upload an image of a potato leaf and predicts the disease.
- **Light Theme:** Ensures a user-friendly and clean interface.

## Installation

### **1. Clone the Repository**

```sh
git clone https://github.com/your-repository-link.git
cd potato-disease-prediction
```

### **2. Install Dependencies**

Make sure you have Python 3 installed. Install the required packages using:

```sh
pip install -r requirements.txt
```

### **3. Run the Application**

```sh
streamlit run app.py
```

## File Structure

```
├── app.py                 # Streamlit UI for the application
├── model                  # Directory containing the trained model (.h5 file)
│   ├── potatoes.h5        # Trained model for prediction
├── images                 # Sample images for testing
├── requirements.txt       # Required dependencies
└── README.md              # Project documentation
```

## Usage

1. **Navigate to the Prediction Page** using the sidebar.
2. **Upload an image** of a potato leaf.
3. The system will **process the image and predict** the disease.
4. **View the results** with confidence scores.

## Dependencies

- **Streamlit** (For UI)
- **TensorFlow / Keras** (For Model Loading)
- **NumPy & Pandas** (For Data Handling)

## Future Improvements

- Add more disease classifications.
- Improve UI with dark/light mode toggle.
- Deploy using Streamlit Sharing or AWS.

## Contributing

If you find any issues or want to improve the project, feel free to create a pull request.



