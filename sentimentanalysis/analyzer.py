import joblib
import streamlit as st
import pandas as pd
import plotly.express as px



path = 'C:/Users/Aditee/OneDrive/Documents/GitHub/Sentiment-Analysis/sentimentanalysis/models.p'
with open(path, 'rb') as joblibmodel:
    data = joblib.load(joblibmodel)
model = data['model']
vectorizer = data['vectorizer']

userinputtext = st.text_area('Enter your text', key='name', height=15)

test_feature = vectorizer.transform([userinputtext])


if st.button('Analyse'):
    if len(userinputtext) > 0:
        ans= model.predict(test_feature)
        st.markdown(ans)
    else:
        st.text('Invalid input')

st.title('Sentiment Analysis using ML')
st.markdown('This application is about analysing sentiments of texts.')

st.sidebar.title('Sentiment Analysis')

st.sidebar.subheader('Actions')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, usecols=['content'])
    if st.checkbox('Show Data'):
        st.write(data.head(8))

    test_file = vectorizer.transform(data["content"])
    csv_prediction = model.predict(test_file)
    df_csv= pd.DataFrame({'Head':csv_prediction})

    st.write(csv_prediction)
    select=st.sidebar.selectbox('Visualisation of Data',['Histogram', 'Pie Chart'], key=1)

    sentiment_visualise = df_csv['Head'].value_counts()
    sentiment_visualise = pd.DataFrame({'Sentiment':sentiment_visualise.index,'Text':sentiment_visualise})
    # st.markdown('### Sentiment count')

    if select == 'Histogram':
        fig = px.bar(sentiment_visualise, x='Sentiment', y='Text', color='Text', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_visualise, values='Text', names='Sentiment')
        st.plotly_chart(fig)

