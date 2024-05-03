import streamlit as st
from keras.models import load_model
import numpy as np

# Load model and labels
model = load_model("model.h5")
labels = np.load("labels.npy")

# Custom CSS for title background and font
st.markdown(
    """
    <style>
    .title {
        background-color: #171717;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-family: 'Times New Roman', Times, serif; /* Added Times New Roman */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the title with custom background
st.markdown('<h1 class="title">Welcome to the flower prediction app</h1>', unsafe_allow_html=True)

# Adding hints in the placeholder parameter of number_input
a = float(st.number_input("sepal length in cm", help="Enter a value between 1 and 8"))
b = float(st.number_input("sepal width in cm",  help="Enter a value between 1 and 5"))
c = float(st.number_input("petal length in cm", help="Enter a value between 1 and 8"))
d = float(st.number_input("petal width in cm",  help="Enter a value between 0.1 and 3"))
btn = st.button("predict")

if btn:
    # Predict the class
    pred = model.predict(np.array([[a, b, c, d]]))  # Ensure correct input shape
    pred_class = np.argmax(pred)
    pred_label = labels[pred_class]
    st.subheader(f"Predicted: {pred_label}")

    # Display the corresponding image
    if pred_class == 0:
        st.image("setosa.jpg", caption='Iris Setosa')
    elif pred_class == 1:
        st.image("versicolor.jpg", caption='Iris Versicolour')
    else:
        st.image("virginica.jpg", caption='Iris Virginica')  # Ensure filename is correct

# Footer
st.markdown("---")
st.markdown("<b style='color: black;'>Developed by Samrat Paul</b><br>Email address: samrat16.sp@gmail.com", unsafe_allow_html=True)
