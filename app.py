import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 64x64 (adjust size based on model's training input)
    image = image.resize((64, 64))
    # Convert to numpy array, normalize, and flatten
    image_array = np.array(image)
    image_flat = image_array.flatten().reshape(1, -1)  # Flatten to 1D array for scikit-learn
    return image_flat

# Function to set background and styling
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #f9e1e1, #ffffff); /* Inspired by Vietnamese flag colors */
        }
        .title {
            font-size: 2.5em;
            color: #c0392b; /* Red from Vietnamese flag */
            text-align: center;
            margin-bottom: 10px;
            font-family: 'Arial', sans-serif;
        }
        .subtitle {
            font-size: 1.2em;
            color: #34495e;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #c0392b;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #a93226;
        }
        .sidebar .sidebar-content {
            background-color: #fffde7; /* Light yellow from Vietnamese flag */
        }
        .stAlert {
            border-radius: 10px;
        }
        .prediction-box {
            background-color: #ffffff;
            border: 2px solid #c0392b;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            font-size: 0.9em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit app
def main():
    set_background()

    # Header with title and subtitle in Vietnamese
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown('<div class="title">Hệ Thống Phát Hiện Viêm Phổi</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Tải lên hình ảnh X-quang để phát hiện viêm phổi hoặc tình trạng bình thường</div>', unsafe_allow_html=True)

    # Sidebar with instructions in Vietnamese
    with st.sidebar:
        st.header("Hướng Dẫn Sử Dụng")
        st.markdown("""
        1. **Tải Hình Ảnh Lên**: Nhấn vào nút chọn file để tải hình ảnh X-quang (JPG, PNG, hoặc JPEG).
        2. **Xem Kết Quả**: Hệ thống sẽ phân tích hình ảnh và hiển thị kết quả dự đoán.
        3. **Độ Tin Cậy**: Nếu có, điểm độ tin cậy sẽ được hiển thị.
        **Lưu Ý**: Đảm bảo hình ảnh là X-quang ngực rõ nét để có kết quả chính xác.
        """)
        st.markdown("---")
        st.subheader("Giới Thiệu")
        st.markdown("Ứng dụng này sử dụng mô hình học máy được huấn luyện sẵn để phát hiện viêm phổi từ hình ảnh X-quang ngực.")

    # Main content
    st.markdown("### Tải Lên Hình Ảnh X-Quang")
    uploaded_file = st.file_uploader("Chọn hình ảnh X-quang...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Hình Ảnh X-Quang Đã Tải Lên", use_container_width=True, clamp=True)

        # Process button
        if st.button("Phân Tích Hình Ảnh"):
            with st.spinner("Đang phân tích hình ảnh..."):
                try:
                    # Load the pre-trained scikit-learn model
                    with open("pca_lr_pipeline.pkl", "rb") as file:
                        model = pickle.load(file)

                    # Preprocess the image
                    processed_image = preprocess_image(image)

                    # Make prediction
                    prediction = model.predict(processed_image)[0]
                    confidence = ""
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(processed_image)[0]
                        pneumonia_prob = prob[1]
                        confidence = f" (Độ Tin Cậy: {pneumonia_prob:.2%})" if prediction == 1 else f" (Độ Tin Cậy: {1 - pneumonia_prob:.2%})"

                    # Display result in a styled box
                    result = "Viêm Phổi" if prediction == 1 else "Bình Thường"
                    st.markdown(
                        f'<div class="prediction-box"><strong>Dự Đoán: {result}</strong>{confidence}</div>',
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Lỗi: {str(e)}. Vui lòng đảm bảo file mô hình và hình ảnh hợp lệ.")

    # Footer
    st.markdown('<div class="footer">© 2025 Hệ Thống Phát Hiện Viêm Phổi.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()