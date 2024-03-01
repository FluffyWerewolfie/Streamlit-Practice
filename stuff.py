import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

st.write("""
# My first app
Sure is nice to *try* new things out!
""")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
df = pd.read_csv("data/Student_Marks2.csv")
print(df.head())

main1,main2 = st.tabs(["Student Marks", "Extra"])
                    
tab1, tab2, tab3,tab4 = main1.tabs(["Study To Marks", "Course Count To Marks","Prediction","Raw Data"])

tab1.subheader("Is there a relationship between studying time and marks got?")
tab1.scatter_chart(df,x='time_study',y='Marks')

tab2.subheader("Is there a relationship between courses attended and marks got?")
tab2.scatter_chart(df,x='number_courses',y='Marks')



tab3.subheader("Study time to Marks")
model.fit(df[['time_study']],df[['Marks']])
y_preds = model.predict(df[['time_study']].sort_values(by='time_study', ascending=True))
tab3.line_chart(y_preds)
tab3.subheader("Number of courses to Marks")
model.fit(df[['number_courses']],df[['Marks']])
y_preds = model.predict(df[['number_courses']].sort_values(by='number_courses', ascending=True))
tab3.line_chart(y_preds)
                        

tab4.subheader("Numbers are superior")
tab4.write(df)


statFile = pd.read_csv("data/tbl_statistics.csv")

main2.scatter_chart(statFile)