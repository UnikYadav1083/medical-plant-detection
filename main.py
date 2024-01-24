import joblib
import streamlit as st
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

feature_extractor = load_model('feature_extractor_model.h5')

import pickle

with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# RF_model = joblib.load('random_forest_model.joblib')

SIZE = 256

with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def predict_image_label(img_path):
    img = preprocess_image(img_path)

    features = feature_extractor.predict(img)

    prediction_RF = loaded_model.predict(features)[0]
    predicted_label = le.inverse_transform([prediction_RF])[0]

    return predicted_label

def main():
    st.title("Image Classification with RandomForest")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        st.write("")
        st.write("Classifying...")

        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Predict"):
            predicted_label = predict_image_label("uploaded_image.jpg")

            st.write("Prediction:")
            st.write(f"The image is classified as: {predicted_label}")

            # Use CSS to set the width of the image
            st.markdown(
                f'<style>img {{ max-width: 70%; width: 100px; height: 300px; }}</style>',
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()





















# import streamlit as st
# import numpy as np
# import cv2
# import pickle
# import joblib
# from sklearn.decomposition import PCA
# from tensorflow.keras.models import load_model
#
# # Load the Xception feature extractor
# feature_extractor = load_model('xception_feature_extractor.h5')
#
# # Load the trained RandomForest model
# RF_model = joblib.load('random_forest_model.joblib')
#
# # Load the class names
# with open('class_names.pkl', 'rb') as file:
#     class_names = pickle.load(file)
#
# def preprocess_image(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (256, 256))  # Assuming your model was trained with images of size 256x256
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.expand_dims(img, axis=0) / 255.0
#     return img
#
# def predict_image_label(img_path, pca, rf_model):
#     img = preprocess_image(img_path)
#     features = feature_extractor.predict(img)
#
#     # Apply PCA
#     features_pca = pca.transform(features)
#
#     # Now predict using the trained RF model.
#     prediction_RF = rf_model.predict(features_pca)[0]
#     predicted_label = class_names[prediction_RF]
#
#     return predicted_label
#
# def main():
#     st.title("Medical Plants Classifier with RandomForest")
#
#     uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#
#     if uploaded_file is not None:
#         st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
#
#         st.write("")
#         st.write("Classifying...")
#
#         with open("uploaded_image.jpg", "wb") as f:
#             f.write(uploaded_file.getbuffer())
#
#         if st.button("Predict"):
#             # You need to replace the following line with your PCA model and training data
#             # For this example, I'm assuming you have already trained the PCA model
#             pca = PCA(n_components=128)  # Replace with your actual PCA model
#             # Assuming you have access to the original training data (replace with actual data)
#             train_data = ...
#
#             # Fit PCA on the features extracted by Xception for the training dataset
#             X_train_pca = pca.fit_transform(train_data)
#
#             predicted_label = predict_image_label("uploaded_image.jpg", pca, RF_model)
#
#             st.write("Prediction:")
#             st.write(f"The image is classified as: {predicted_label}")
#
# if __name__ == "__main__":
#     main()
#
#
#
#
#
#
#
#
#
#











# import streamlit as st
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# import pickle
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
#
# # Load the feature extractor model
# feature_extractor = load_model('feature_extractor_model.h5')
#
# # Load the trained Random Forest model
# with open('random_forest_model.pkl', 'rb') as model_file:
#     RF_model = pickle.load(model_file)
#
# SIZE = 256
#
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (SIZE, SIZE))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.expand_dims(img, axis=0) / 255.0
#     return img
#
# def predict_image_label(image_path):
#     img = preprocess_image(image_path)
#
#     # Extract features using the feature extractor model
#     features = feature_extractor.predict(img)
#
#     # Predict using the trained RandomForest model
#     prediction_RF = RF_model.predict(features)[0]
#     predicted_label = le.inverse_transform([prediction_RF])[0]
#
#     return predicted_label
#
# def main():
#     st.title("Image Classification with RandomForest and Streamlit")
#
#     uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#
#     if uploaded_file is not None:
#         st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
#         st.write("")
#         st.write("Classifying...")
#
#         # Save the uploaded file
#         with open("uploaded_image.jpg", "wb") as f:
#             f.write(uploaded_file.getbuffer())
#
#         # Get the predicted label
#         predicted_label = predict_image_label("uploaded_image.jpg")
#
#         st.write("Prediction:")
#         st.write(f"The image is classified as: {predicted_label}")
#
# if __name__ == "__main__":
#     main()














# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import joblib
# from PIL import Image
# import sklearn
#
# # Check scikit-learn version
# sklearn_version = sklearn.__version__
# st.write(f"Using scikit-learn version: {sklearn_version}")
#
# # Ensure compatibility with the scikit-learn version used during model training
# required_sklearn_version = "1.3.2"  # Replace with the version used during training
#
# if sklearn_version != required_sklearn_version:
#     st.error(f"Error: Incompatible scikit-learn version. Expected version {required_sklearn_version}.")
#     st.stop()
#
# # Load Xception feature extractor and RandomForest model
# xception_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
# random_forest_model = joblib.load('random_forest_model.pkl')
# class_names = joblib.load('class_names.pkl')
#
# # Streamlit App
# st.title("Image Classification Web App")
#
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#
# if uploaded_file is not None:
#     # Preprocess the image
#     img = Image.open(uploaded_file)
#     img = img.resize((299, 299))  # Xception input size
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.keras.applications.xception.preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)
#
#     # Extract Xception features
#     features = xception_model.predict(img_array)
#
#     # Make prediction using RandomForest model
#     prediction = random_forest_model.predict(features)
#
#     # Display the results
#     st.image(img, caption="Uploaded Image", use_column_width=True)
#     st.write("Class Prediction:", class_names[prediction[0]])
