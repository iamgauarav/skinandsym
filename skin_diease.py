
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os

# Load the saved model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("skin_cancer_resnet50.pth", map_location=torch.device('cpu')))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict(image):
    img_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        if predicted == 0:
            prediction = 'Melanoma'
        else:
            prediction = 'Allergy'
    return prediction

def main():
    st.title('Skin Cancer Prediction App')
    st.write('Upload an image of the skin lesion to predict whether it is melanoma or an allergy.')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            prediction = predict(image)
            st.success(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
