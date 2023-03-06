import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import joblib

from PIL  import Image

pipeline_model = joblib.load(open('RandomForest.pkl', 'rb'))


#def predict_disease(docx):
    #words = []
   # for word in docx:
       # words.append(word)
      #  pred = pipeline_model.predict(words)

   # return pred

    #results = pipeline_model.predict(docx)
    #return results[0]


def main():
    st.title('Disease Prediction')
    menu = ['Home', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Disease Prediction')

        with st.form(key='disease_clf_form'):
            raw_text = st.text_area('Type Symptomns Here!')
            submit_text = st.form_submit_button(label='Submit')
        
        if submit_text:
            col1, col2 = st.columns(2)

            #prediction = predict_disease(raw_text)

            with col1:
                st.success('Symptomns')
                st.write(raw_text)

            with col2:
                st.success('Predictions')
                #prediction = pipeline_model.predict(word for word in raw_text)

                words = []

                for word in raw_text:
                    words.append(word)

                pred = pipeline_model.predict(words)

                st.write(pred[0])

    else:
        st.subheader('About')




if __name__ == '__main__':
    main()