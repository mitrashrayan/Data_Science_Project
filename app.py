
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv("C:\\Users\\LENOVO\\PycharmProjects\\pythonProject\\Admission_Prediction.csv")
def fill_missing(column):
   df[column] = df[column].fillna(df[column].mean())
missing_column = ['GRE Score', 'TOEFL Score', 'University Rating']
for column in missing_column:
    fill_missing(column)

df.drop(columns=['Serial No.'], inplace=True)
x = df.drop(columns=['Chance of Admit'])
y = df['Chance of Admit']
scaler = StandardScaler()


st.title('Admission Data Science Project')
GRE_Score = st.number_input('GRE_Score')
TOEFL_Score = st.number_input('TOEFL Score')
University_Rating = st.number_input('University Rating')
SOP = st.number_input('SOP', min_value= 0 , max_value=5)
LOR = st.number_input('LOR')
CGPA = st.number_input('CGPA')
Research = st.number_input('Research')

user_input= [[GRE_Score, TOEFL_Score, University_Rating, SOP,LOR,CGPA,Research]]
scaler.fit(x)
scaled_user_input= scaler.transform(user_input)
#st.write(user_input)
#st.write(scaled_user_input)
loaded_model = pickle.load(open('lr_for_admission', 'rb'))
result = loaded_model.predict(scaled_user_input)


if st.button("Predict"):
    result_percentage = result * 100
    st.header( " percentage of you getting admitted in university is " + str(result_percentage))