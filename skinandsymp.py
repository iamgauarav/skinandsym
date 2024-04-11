import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image

class HealthCareChatBot:
    def __init__(self):
        self.training = pd.read_csv('Training.csv')
        self.testing = pd.read_csv('Testing.csv')
        self.cols = self.training.columns[:-1]
        self.x = self.training[self.cols]
        self.y = self.training['prognosis']
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.y)
        self.y = self.le.transform(self.y)
        self.reduced_data = self.training.groupby(self.training['prognosis']).max()
        self.description_list = {}
        self.severityDictionary = {}
        self.precautionDictionary = {}
        self.symptoms_dict = {}
        self.clf1 = DecisionTreeClassifier()

    def load_data(self):
        self.getSeverityDict()
        self.getDescription()
        self.getprecautionDict()

    def load_decisiontree(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        clf = self.clf1.fit(self.x_train, self.y_train)
        return clf
    
    def load_cols(self):
        column = self.cols
        return column
 
    # Other methods remain the same

    def train_model(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=8)
        self.model = self.build_ann_model()
        self.model.fit(self.x_train, self.y_train, epochs=7, batch_size=8)

    def build_ann_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.x_train.shape[1], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(len(np.unique(self.y_train)), activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def predict_using_ann(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print(f"Accuracy on test data: {accuracy}")

    def getDescription(self):
        with open('symptom_Description.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                _description = {row[0]: row[1]}
                self.description_list.update(_description)

    def getSeverityDict(self):
        with open('Symptom_severity.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            try:
                for row in csv_reader:
                    symptom = row[0]
                    severity_str = row[1]
                    try:
                        severity = int(severity_str)
                        self.severityDictionary[symptom] = severity
                    except ValueError:
                        print(f"Invalid severity value '{severity_str}' for symptom '{symptom}'. Skipping...")
            except IndexError:
                pass

    def getprecautionDict(self):
        with open('symptom_precaution.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
                self.precautionDictionary.update(_prec)

    def check_pattern(self, dis_list, inp):
        pred_list = []
        inp = inp.replace(' ', '_')
        patt = f"{inp}"
        regexp = re.compile(patt)
        pred_list = [item for item in dis_list if regexp.search(item)]
        if len(pred_list) > 0:
            return 1, pred_list
        else:
            return 0, []

    def sec_predict(self, symptoms_exp):
        df = pd.read_csv('Training.csv')
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[[symptoms_dict[item]]] = 1
        return rf_clf.predict([input_vector])

    def print_disease(self, node):
        node = node[0]
        val = node.nonzero()
        disease = self.le.inverse_transform(val[0])
        return list(map(lambda x: x.strip(), list(disease)))

    def calc_condition(self, exp, days):
        sum = 0
        for item in exp:
            sum = sum + self.severityDictionary[item]
        if (sum * days) / (len(exp) + 1) > 13:
            st.write("You should take the consultation from a doctor.")
        else:
            st.write("It might not be that bad, but you should take precautions.")

    def tree_to_code(self, tree, feature_names):
        tree_ = tree.tree_
        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" 
                        for i in tree_.feature]
        chk_dis = ",".join(feature_names).split(",")
        symptoms_present = []

        disease_input = st.text_input("\nEnter the symptom you are experiencing: ")
        conf, cnf_dis = self.check_pattern(chk_dis, disease_input)
        if conf == 1:
            st.write("Searches related to input:")
            for num, it in enumerate(cnf_dis):
                st.write(num, ")", it)
            if num != 0:
                conf_inp = st.selectbox(f"Select the one you meant (0 - {num}):", list(range(num + 1)))
            else:
                conf_inp = 0
                st.write("Select the one you meant: 0")
            disease_input = cnf_dis[conf_inp]
        else:
            st.write("Enter a valid symptom.")

        num_days = st.number_input("Okay. For how many days?", min_value=0, step=1)

        def recurse(node, depth):
    
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                if name == disease_input:
                    val = 1
                else:
                    val = 0
                if val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = self.print_disease(tree_.value[node])
                red_cols = self.reduced_data.columns
                symptoms_given = red_cols[self.reduced_data.loc[present_disease].values[0].nonzero()]

                print("Are you experiencing any ")
                symptoms_exp=[]
                # for syms in enumerate(symptoms_given):    # Generate unique key using both symptom name and index
                #     inp = st.text_input(f"Are you experiencing any {syms}?", "")
                #     while inp not in ["yes", "no"]:
                #         st.write("Please provide a proper answer (yes/no): ")
                #         inp2 = st.text_input(f"Are you experiencing any {syms}?", "")
                #         inp=inp2
                #     if inp == "yes":
                #         symptoms_exp.append(syms)

                for syms in symptoms_given:
                    inp =  st.radio(f"Are you experiencing any {syms}?", options=["yes", "no"],index=1)
                    if inp == "yes":
                        symptoms_exp.append(syms)

                second_prediction = self.sec_predict(symptoms_exp)
                self.calc_condition(symptoms_exp, num_days)
                if present_disease[0] == second_prediction[0]:
                    st.write("You may have ", present_disease[0])
                    st.write(self.description_list[present_disease[0]])
                else:
                    st.write("You may have ", present_disease[0], "or ", second_prediction[0])
                    st.write(self.description_list[present_disease[0]])
                    st.write(self.description_list[second_prediction[0]])
                precution_list = self.precautionDictionary[present_disease[0]]
                st.write("Take following measures : ")
                for  i,j in enumerate(precution_list):
                    st.write(i+1,")",j)
        recurse(0,0)

def generate_response(input_text):
    # This is a dummy response generation function
    # You can replace it with a more sophisticated chatbot model
    return "I received: " + input_text

def display_healthcare_chatbot():
    bot = HealthCareChatBot()
    bot.load_data()
    tree = bot.load_decisiontree()
    feature_names = bot.load_cols()
    bot.tree_to_code(tree, feature_names)

def predict(image):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("skin_cancer_resnet50.pth", map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        if predicted == 0:
            prediction = 'Melanoma'
        else:
            prediction = 'Allergy'
    return prediction

def display_skin_cancer_app():
    st.title('Skin Disease prediction ')
    st.write('Upload an image of the skin lesion to predict whether it is melanoma or an allergy.')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            prediction = predict(image)
            st.success(f'Prediction: {prediction}')

def main():
    # Set page config to add CSS styles
    st.set_page_config(
        page_title="Healthcare Application",
        page_icon=":hospital:",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Add CSS styles
    st.write(
        """
        
<style>
    /* Body Styles */
    body {
        font-family: 'Roboto', sans-serif; /* Google Font */
        color: #333; /* Text color */
        margin: 0; /* Remove default margin */
        padding: 0; /* Remove default padding */
        background-image: url('https://source.unsplash.com/random/1920x1080/?nature,landscape');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Application Container Styles */
    .stApp {
        max-width: 1200px;
        margin: auto;
        padding: 40px;
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white background */
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); /* Drop shadow effect */
        overflow: hidden; /* Hide content overflow */
    }

    /* Title Styles */
    .stTitle {
        font-size: 2.5rem;
        margin-bottom: 20px;
    }

    /* Subtitle Styles */
    .stSubtitle {
        font-size: 1.5rem;
        margin-bottom: 10px;
    }

    /* Button Styles */
    .stButton {
        background-color: #4CAF50; /* Green background color */
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
        font-size: 1rem;
    }

    .stButton:hover {
        background-color: #45a049; /* Darker green on hover */
    }

    /* Form Input Styles */
    .stTextInput, .stSelectbox, .stTextArea {
        border: 1px solid #ccc; /* Gray border */
        border-radius: 5px;
        padding: 12px;
        margin-bottom: 20px;
        width: 100%;
        box-sizing: border-box; /* Include padding and border in element's total width */
        font-size: 1rem;
    }

    .stTextInput:focus, .stSelectbox:focus, .stTextArea:focus {
        border-color: #4CAF50; /* Green border color when focused */
    }

    /* Image Styles */
    .stImage {
        max-width: 100%;
        height: auto;
        border-radius: 5px;
        margin-bottom: 20px;
    }

    /* Card Styles */
    .stCard {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    }
</style>
        """,
        unsafe_allow_html=True
    )

    # Display the title and greeting
    st.title("Choose the Application to Open")
    st.write("Select which application you want to open:")
    selected_option = st.radio("Select Application:", ("Common Diseases", "Skin Diseases"), None)

    name = st.text_input("What is your name?")
    greeting = f"Hello {name}!" if name else ""
    st.write(greeting)

    # Application selection

    # Display the selected application
    if selected_option == "Common Diseases":
        display_healthcare_chatbot()
    elif selected_option == "Skin Diseases":
        display_skin_cancer_app()

if __name__ == "__main__":
    main()
