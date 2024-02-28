import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

st.write("""
# My first app
Hello *world!*
""")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
df = pd.read_csv("data/Student_Marks2.csv")
print(df.head())



m_score=model.score(df[['time_study']],df[['Marks']])
                        
print(m_score)

                    
tab1, tab2, tab3,tab4 = st.tabs(["Study To Marks", "Course Count To Marks","Prediction","Raw Data"])

tab1.subheader("Is there a relationship between studying time and marks got?")
tab1.scatter_chart(df,x='time_study',y='Marks')

tab2.subheader("Is there a relationship between courses attended and marks got?")
tab2.scatter_chart(df,x='number_courses',y='Marks')



tab3.subheader("Is there a relationship between courses attended and marks got?")
model.fit(df[['time_study'],df[['Marks']])
y_preds = model.predict(df[['time_study'])
tab3.line_chart(y_preds)

model.fit(df[['number_courses'],df[['Marks']])
y_preds = model.predict(df[['number_courses'])
tab3.line_chart(y_preds)
                        

tab4.subheader("Numbers are superior")
tab4.write(df)
