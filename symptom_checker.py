import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
from tensorflow.keras.models import load_model



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
        column=self.cols
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
                _description={row[0]:row[1]}
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
                _prec={row[0]:[row[1],row[2],row[3],row[4]]}
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

    def sec_predict(self,symptoms_exp):
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


        disease_input=st.text_input("\nEnter the symptom you are experiencing: ")
        conf, cnf_dis = self.check_pattern(chk_dis, disease_input)
        if conf == 1:
            st.write("Searches related to input:")
            for num, it in enumerate(cnf_dis):
                st.write(num, ")", it)
            if num!=0:
                conf_inp = st.selectbox(f"Select the one you meant (0 - {num}):", list(range(num + 1)))
            else:
                conf_inp=0
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

# Streamlit UI
def main():
   
   training = pd.read_csv('Training.csv')
   testing= pd.read_csv('Testing.csv')
   cols= training.columns
   cols= cols[:-1]
   x = training[cols]
   y = training['prognosis']
   y1= y
   reduced_data = training.groupby(training['prognosis']).max()

    #mapping strings to numbers
   le = preprocessing.LabelEncoder()
   le.fit(y)
   y = le.transform(y)
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
   testx    = testing[cols]
   testy    = testing['prognosis']  
   testy    = le.transform(testy)
   clf1  = DecisionTreeClassifier()
   clf = clf1.fit(x_train,y_train)
   st.title("HealthCare ChatBot")

    # Initialize the HealthCareChatBot instance
   bot = HealthCareChatBot()
   bot.load_data()

    # Get user in
   st.text_input("You:", "")
    # Assume you have the decision tree and feature names accessible
   tree=bot.load_decisiontree()
   feature_names = bot.load_cols()

        # Call the tree_to_code method
   bot.tree_to_code(tree, feature_names)

       
   
if __name__ == "__main__":
    main()
