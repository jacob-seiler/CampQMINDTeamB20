import streamlit as st

def image_selector():
    uploaded_file = st.file_uploader(label='Upload a custom image')
    
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)

def main():
    st.title('Fashion Classifier')
    image_selector()

if __name__ == '__main__':
    main()