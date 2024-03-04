import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import PolynomialFeatures
TF_ENABLE_ONEDNN_OPTS=0

st.write("""
# My first app
Sure is nice to *try* new things out!
""")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
df = pd.read_csv("data/Student_Marks2.csv")
##print(df.head())

main1,main2,main3 = st.tabs(["Student Marks", "Web App Y1 Statistics","TensorFlow Shenanigans"])
                    
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
webTraffic = statFile.drop(['statisticID','Device','ScreenHeight','ScreenWidth','refferal','statTimeStamp'],axis=1)
main2.line_chart(webTraffic)


TFtab1, TFtab2,TFtab3 = main3.tabs(["Pretrained Neural Network", "Visualize A Network Train","Raw Dataset"])
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

TFData = pd.read_csv("data/training1.csv")

TFtab3.subheader("100 entries per row making a 10x10 grid image with 0 for a blank pixel 1 for a colored pixel:\n239 Labeled Examples")
TFtab3.write(TFData)

y=TFData['label']
X = pd.get_dummies(TFData.drop(['entryID','label'], axis=1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.6)

predictBtnPlaceholder=TFtab1.empty()
TFPredictCol1,TFPredictCol2 = TFtab1.columns(2)
TFPredictCol1.subheader('Original Label:')
TFPredictCol2.subheader('Network Guess:')
LabelList = TFPredictCol1.empty()
PredictionList = TFPredictCol2.empty()


if predictBtnPlaceholder.button('Predict'):
    TFmodel = tf.keras.models.load_model("Hellla Experimental1",compile=False)
    #TFmodel = tf.keras.models.load_model("Hellla Experimental.keras",compile=False)
    y_prediction = TFmodel.predict(X)
    y_prediction=np.round(y_prediction,0)
    LabelList.write(y)
    PredictionList.write(y_prediction)
    TFtab1.subheader('Accuracy: ' + str(accuracy_score(y,y_prediction)))

trainBtnPlaceholder = TFtab2.empty()
TFcol1,TFcol2 = TFtab2.columns(2)
chart1 = TFcol1.empty()
chart2 = TFcol2.empty()
if trainBtnPlaceholder.button('Train'):

    TFmodel= Sequential()
    TFmodel.add(Dense(units=100,activation='relu',input_dim=len(X_train.columns)))
    TFmodel.add(tf.keras.layers.Dropout(.2, input_shape=(100,)))
    TFmodel.add(Dense(units=10,activation='sigmoid'))
    TFmodel.add(Dense(units=100,activation='relu'))
    TFmodel.add(Dense(units=10,activation='sigmoid'))
    TFmodel.add(Dense(units=1,activation='relu'))

    TFmodel.compile(loss=tf.keras.losses.MeanAbsoluteError(),optimizer='adam',metrics=[tf.keras.metrics.Accuracy()])

    dataFun = []
    dataFun2=[]
    TFcol1.subheader('Accuracy On Trained Data')
    lol = chart1.line_chart(dataFun)
    TFcol2.subheader('Accuracy On Unseen Data')
    lol2 = chart2.line_chart(dataFun2)

    for x in range(100):
        TFmodel.fit(X_train, y_train, epochs=20)
        y_prediction = TFmodel.predict(X_train)
        y_prediction=np.round(y_prediction,0)
        dataFun.append(accuracy_score(y_train,y_prediction))

        y_prediction = TFmodel.predict(X_test)
        y_prediction=np.round(y_prediction,0)
        dataFun2.append(accuracy_score(y_test,y_prediction))
        lol.add_rows(dataFun)
        lol2.add_rows(dataFun2)
    #TFmodel.save('Hellla Experimental1')
    #TFmodel.save('Hellla Experimental.keras')

