import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from PIL import Image
import streamlit.components.v1 as components
# import matplotlib.pyplot as plt
import pickle
# load model 
import joblib
import bz2
import pickle
import _pickle as cPickle
from extract import getBagian, extract_pasal
# [theme]
# base="light"
# primaryColor="purple"

st.set_page_config(layout='wide')

def main():
    menu = ["Verdict Prediction","Dispute Prediction","Metric Scores"]
    
    choice = st.sidebar.selectbox("Select Menu", menu)

        
    if choice == "Verdict Prediction":
        # st.subheader("Prediction from Model")
#         st.title("MachineLearning Analytics App")
        st.subheader("Verdict Prediction")
        # iris= Image.open('iris.png')

        model= open("vmodel.pkl", "rb")
        knn=joblib.load(model)

        data = st.file_uploader('Upload File PDF',type='.pdf')
        if data == None:
            st.write('Silakan Upload File dengan format pdf')

        if st.button('Extract Features'):
            # aDict = pickle.load(open("aDict.p","rb"))
            text = getBagian(data)
            pasal = extract_pasal(text)
            df = pd.DataFrame.from_dict(pasal)
            pslist = df.pasal.unique()
            st.dataframe(pslist)
            df[['pasal','drop']] = df['pasal'].str.split(' ayat',expand=True)
            st.dataframe(df.pasal.unique())
        st.subheader("Features")
        #Intializing
        c1,c2 = st.beta_columns((1,1))
        with c1:
            sl = st.number_input(label="FP Lengkap",value=1,min_value=0, max_value=1, step=1)
            sw = st.number_input(label="FP Tepat Waktu",value=1,min_value=0, max_value=1, step=1)
            pl = st.number_input(label="Keterangan FP Sesuai",value=0,min_value=0, max_value=1, step=1)
            dm1 = st.number_input(label="FP Diganti Dibatalkan",value=1,min_value=0, max_value=1, step=1)
            dm2 = st.number_input(label="FP Tidak Double Kredit",value=1,min_value=0, max_value=1, step=1)
        with c2:
            dm0 = st.number_input(label="Lawan PKP",value=1,min_value=0, max_value=1, step=1)
            dm3 = st.number_input(label="Lawan Disanksi",value=1,min_value=0, max_value=1, step=1)
            dm4 = st.number_input(label="Lawan Lapor",value=1,min_value=0, max_value=1, step=1)
            dm5 = st.number_input(label="Minta Tanggung Jawab Lawan",value=1,min_value=0, max_value=1, step=1)
            pw = st.number_input(label="PPN telah dibayar",value=0,min_value=0, max_value=1, step=1)

        if st.button("Click Here to Classify"):
            dfvalues = pd.DataFrame(list(zip([sl],[sw],[pl],[pw])),columns =['lengkap', 'tepatwaktu', 'ketsesuai', 'adapembayaran'])
            input_variables = np.array(dfvalues[['lengkap', 'tepatwaktu', 'ketsesuai', 'adapembayaran']])
            prediction = knn.predict(input_variables)
            if prediction == 'ditolak':
                st.subheader('Prediksi Hasil Verdict')
                st.title('Permohonan Banding Ditolak')
            elif prediction =='sebagian':
                st.subheader('Prediksi Hasil Verdict')
                st.title('Permohonan Banding Diterima Sebagian')
            else:
                st.subheader('Prediksi Hasil Verdict')
                st.title('Permohonan Banding Diterima Seluruhnya')
    
    elif choice == "Dispute Prediction":
        st.title("Dispute Predicction")

    elif choice == "Metric Scores":
        st.title("Metric Scores")
        

if __name__=='__main__':
    main()
