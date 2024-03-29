import streamlit as st
from streamlit_image_select import image_select
import keras
from keras import Sequential
from PIL import Image
from io import BytesIO
import numpy as np
from rembg import remove
import urllib.request
import os

REMOTE_URL = "https://raw.githubusercontent.com/jacob-seiler/CampQMINDTeamB20/main/FashionClassifierUI"
IMAGE_PATHS = [
            "images/custom.jpg",
            "images/t-shirt.jpg",
            "images/pants.jpg",
            "images/pullover.jpg",
            "images/dress.jpg",
            "images/coat.jpg",
            "images/sandal.jpg",
            "images/shirt.jpg",
            "images/sneaker.jpg",
            "images/bag.jpg",
            "images/boots.jpg"
        ]

@st.experimental_singleton
def load_model() -> Sequential:
    if not os.path.isdir("model"):
        os.makedirs("model/variables")

        for file in ["variables/variables.data-00000-of-00001", "variables/variables.index", "keras_metadata.pb", "saved_model.pb"]:
            urllib.request.urlretrieve(f'{REMOTE_URL}/model/{file}', f"model/{file}")
    
    return keras.models.load_model("model")

def image_selector():
    selection = image_select(
        label="Sample images",
        images=IMAGE_PATHS,
        captions=["Upload Custom", *classes],
        return_value="index"
    )

    img = None

    if selection == 0:
        file = st.file_uploader(label="Upload a custom image")

        if file is not None:
            bytes_data = file.read()
            img = Image.open(BytesIO(bytes_data))
    else:
        img = Image.open(IMAGE_PATHS[selection])

    if img is not None:
        predictor(img)

def image_uploader():
    uploaded_file = st.file_uploader(label='Upload a custom image')

    if uploaded_file is not None:
        # image_data = uploaded_file.getvalue()

        bytes_data = uploaded_file.read()
        image = Image.open(BytesIO(bytes_data))
        predictor(image)

def predictor(image):
    st.image(image)

    with st.spinner("Predicting..."):
        # Process image
        image = remove(image) # Remove background
        image = image.convert("L") # Make greyscale
        image = image.resize((28, 28)) # Resize

        # Convert to numpy array
        pred_x = np.array([np.asarray(image)])

        # Format input
        pred_x = pred_x.reshape((pred_x.shape[0], 28, 28, 1))
        pred_x = pred_x.astype("float32")
        pred_x = pred_x / 255.0
        pred_x = np.reshape(pred_x, (pred_x.shape[0], pred_x.shape[1], pred_x.shape[2], 1))
        
        # Run prediction on model
        pred_y = np.argmax(model.predict(pred_x), axis=-1)
        pred_y = classes[int(pred_y)]
        st.header(pred_y)
        st.balloons()

model = None
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def main():
    with st.spinner("Loading model..."):
        global model
        model = load_model()

    if not os.path.isdir("images"):
        with st.spinner("Loading images..."):
            os.makedirs("images")

            for path in IMAGE_PATHS:
                urllib.request.urlretrieve(f'{REMOTE_URL}/{path}', f"{path}")

    st.title('Fashion Classifier')
    image_selector()

if __name__ == '__main__':
    main()