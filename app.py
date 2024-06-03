import os
import pandas as pd
from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re
import string
from nltk.corpus import stopwords

# Download nltk data
nltk.download('stopwords')
nltk.download('punkt')

# Inisialisasi stop words
stop_words = set(stopwords.words("english"))

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocess clean text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

#  VADER sentiment analysis
def vader_sentiment(text):
    sa = SentimentIntensityAnalyzer()
    scores = sa.polarity_scores(text=text)
    return scores

#  sentiment label
def get_sentiment_label(scores):
    if scores['pos'] > scores['neg'] and scores['pos'] > scores['neu']:
        return "Positive"
    elif scores['neg'] > scores['pos'] and scores['neg'] > scores['neu']:
        return "Negative"
    else:
        return "Neutral"

# Route for the form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text1 = request.form.get('text1')
        processed_doc1 = preprocess_text(text1)
        file = request.files.get('file')
        
        table_data = None
        average_positive = None
        average_neutral = None
        average_negative = None
        average_compound = None
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Baca the CSV file
            df = pd.read_csv(file_path)
            
            # Gunakan nama text untuk text_column
            text_column = 'text'
            
            try:
                # Preprocess text di CSV file
                df['processed_text'] = df[text_column].apply(preprocess_text)
                
                # sentiment analysis CSV
                df['sentiment'] = df['processed_text'].apply(lambda x: vader_sentiment(x))
                
                # Extract sentiment scores
                df['positive'] = df['sentiment'].apply(lambda x: x['pos'])
                df['neutral'] = df['sentiment'].apply(lambda x: x['neu'])
                df['negative'] = df['sentiment'].apply(lambda x: x['neg'])
                df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
                
                # sentiment label
                df['label'] = df['sentiment'].apply(lambda x: get_sentiment_label(x))
                
                table_data = df.drop(columns=['sentiment']).values.tolist()
                headers = df.drop(columns=['sentiment']).columns.tolist()
                table_data.insert(0, headers)

                # Menghitung average sentiment scores for untuk CSV
                average_positive = df['positive'].mean()
                average_neutral = df['neutral'].mean()
                average_negative = df['negative'].mean()
                average_compound = df['compound'].mean()
            except KeyError:
                return f"Error: The specified column '{text_column}' does not exist in the uploaded CSV. Available columns: {df.columns.tolist()}"
        
        # Input User
        sentiment_scores = vader_sentiment(processed_doc1)
        compound = round((1 + sentiment_scores['compound']) / 2, 2)
        average_popularity = round((sentiment_scores['pos'] + sentiment_scores['neg'] + sentiment_scores['neu']) / 3, 2)

        if sentiment_scores['pos'] > sentiment_scores['neg'] and sentiment_scores['pos'] > sentiment_scores['neu']:
            final = "Positive"
        elif sentiment_scores['neg'] > sentiment_scores['pos'] and sentiment_scores['neg'] > sentiment_scores['neu']:
            final = "Negative"
        else:
            final = "Neutral"

        return render_template('form.html',
                               title="Sentiment Analysis",
                               text1=text1,
                               final=final,
                               text2=sentiment_scores['pos'],
                               text3=sentiment_scores['neu'],
                               text5=sentiment_scores['neg'],
                               text4=compound,
                               average_popularity=average_popularity,
                               average_positive=average_positive,
                               average_neutral=average_neutral,
                               average_negative=average_negative,
                               average_compound=average_compound,
                               table_data=table_data)
    
    return render_template('form.html', title="Sentiment Analysis")

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
