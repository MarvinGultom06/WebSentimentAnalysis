import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords


# Download library nltk
nltk.download('stopwords')
nltk.download('punkt')


# Gunakan stopwords untuk menghilangkan kata-kata tidak berguna menggunakan inggris
stop_words = stopwords.words('english')

# Buat instance aplikasi web dengan Flask
app = Flask(__name__)

# menangani apabila file CSV yang akan dibaca tidak ditemukan
try:
  data = pd.read_csv('twitter_data.csv')
except FileNotFoundError:
  print("Error: CSV file not found.")
  exit()

# Fungsi untuk membersihkan teks
def preprocess_text(text):
    # menghapus karakter selain huruf dan angka, dan membuang stopwords.
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Fungsi untuk melakukan Vader sentiment analysis pada sebuah teks
def vader_sentiment(text):
    sa = SentimentIntensityAnalyzer()
    # menggunakan library Vader untuk memberi skor sentimen pada sebuah teks.
    scores = sa.polarity_scores(text=text)
    return scores

# Fungsi untuk menangani halaman utama aplikasi.
@app.route('/')
def my_form():
    if 'text' in data.columns:
        sample_text = data['text'].iloc[0]
    else:
        sample_text = "Error."
    return render_template('form.html', sample_text=sample_text)

# Fungsi untuk menangani form yang disubmit pengguna pada halaman utama.
@app.route('/', methods=['POST'])
def my_form_post():
    text1 = request.form['text1'].lower()
    processed_doc1 = preprocess_text(text1)

    # Vader sentiment analysis
    sentiment_scores = vader_sentiment(processed_doc1)
    compound = round((1 + sentiment_scores['compound']) / 2, 2)
    
    # Analisis atau visualisasi tambahan berdasarkan kumpulan data
    # hitung skor sentimen rata-rata untuk berbagai kategori:
    if 'category' in data.columns:
        category_sentiment = data.groupby('category')['compound'].mean()
        average_sentiment = category_sentiment.mean()
    else:
        average_sentiment = None

    return render_template('form.html',
                           final=compound,
                           text1=processed_doc1,
                           text2=sentiment_scores['pos'],
                           text5=sentiment_scores['neg'],
                           text4=compound,
                           text3=sentiment_scores['neu'],
                           average_sentiment=average_sentiment)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
