import streamlit as st
import pickle
import string
from nltk.stem.porter import PorterStemmer
import requests
from streamlit_lottie import st_lottie

ps = PorterStemmer()

import re

def custom_tokenizer(text):
    # Use regular expression to split on non-alphanumeric characters
    return re.findall(r'\w+', text)


def transform_text(text):
    text = text.lower()
    text = custom_tokenizer(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def remove_stopwords(text):
    stopwords = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'you’re', 'you’ve', 'you’ll',
        'you’d', 'your', 'yours', 'yourself', 'yourselves', 'he', 'hi', 'his', 'himself', 'she', 'she’s',
        'her', 'hers', 'herself', 'it', 'it’s', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'that’ll', 'these', 'those', 'am', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
        't', 'can', 'will', 'just', 'don', 'don’t', 'should', 'should’ve', 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', 'aren’t', 'couldn', 'couldn’t', 'didn', 'didn’t', 'doesn', 'doesn’t', 'hadn',
        'hadn’t', 'hasn', 'hasn’t', 'haven', 'haven’t', 'isn', 'isn’t', 'ma', 'mightn', 'mightn’t', 'mustn',
        'mustn’t', 'needn', 'needn’t', 'shan', 'shan’t', 'shouldn', 'shouldn’t', 'wasn', 'wasn’t', 'weren',
        'weren’t', 'won', 'won’t', 'wouldn', 'wouldn’t', 'in'

    ])

    # Split the text into words
    words = text.split()

    # Filter out stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords]

    # Join the filtered words back into a sentence
    result = ' '.join(filtered_words)

    return result


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://lottie.host/13189996-3072-430b-b3dc-1b0a956a3aa2/hz1gVuph0c.json"
lottie_hello = load_lottieurl(lottie_url_hello)

def st_lottie_with_size(lottie, width, height, key=None):
    st_lottie(lottie, key=key, width=width, height=height)

st_lottie_with_size(lottie_hello, width=300, height=300, key="hi")

if st.button('Predict'):

    # 1. preprocess
    message = remove_stopwords(input_sms)
    transformed_sms = transform_text(message)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


