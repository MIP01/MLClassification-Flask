import os
from flask import Flask, make_response, render_template, request, flash, redirect, url_for
from itertools import zip_longest
import pandas as pd 
import numpy as np
import wordcloud
import pickle
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string 
import re #regex library
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Ambil input teks dari form
    input_text = request.form['input_text']

    # Preprocessing teks
    preprocessed_text = preprocess_text(input_text)

    # Menghapus stopwords
    processed_text_WSW = stopwords_removal(preprocessed_text)

    # Membangun kamus singkatan
    normalizad_word_dict = build_normalizad_word_dict()

    # Normalisasi kata
    text_normalized = normalized_term(processed_text_WSW, normalizad_word_dict)

    # menggabungkan setiap elemen dalam text_normalized menjadi satu string
    string_data = ' '.join(text_normalized)

    # Melakukan stemming
    stemmed_text = stem_text(string_data)

    #error handle jika input kosong atau hanya terdiri dari 1 kata
    if not input_text or len(input_text.split()) < 2:
        flash('Text input is empty or invalid. Please provide a valid input.')
        return redirect(url_for('index'))
    
    #error handle jika processed_text_WSW kosong setelah Menghapus stopwords
    elif len(processed_text_WSW) == 0:
        flash('Input cannot be predicted. Please enter a valid input.')
        return redirect(url_for('index'))

    # prediksi
    predictions, model_names, confidences, total_bullying_confidence, total_non_bullying_confidence, conclusion, accuracy_scores, precision_scores, recall_scores, f1_scores, tmp_filepath, tmp_filepath2, tmp_filepath3 = predict(stemmed_text)

    return render_template('result.html',input_text=input_text, preprocessed_text=stemmed_text, predictions=predictions, model_names=model_names, confidences=confidences, 
                           total_bullying_confidence=total_bullying_confidence, total_non_bullying_confidence=total_non_bullying_confidence, conclusion=conclusion, accuracy_scores=accuracy_scores, 
                           precision_scores=precision_scores, recall_scores=recall_scores, f1_scores=f1_scores, img_path=tmp_filepath, img_path2=tmp_filepath2, img_path3=tmp_filepath3, zip=zip_longest)

def preprocess_text(text):
    # Case folding
    text = text.lower()
    
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')

    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())

    # remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")

    #remove number
    text = re.sub(r"\d+", "", text)

    #remove punctuation
    text = text.translate(str.maketrans("","",string.punctuation))

    #remove whitespace leading & trailing
    text = text.strip()

    #remove multiple whitespace into single whitespace
    text = re.sub('\s+',' ',text)

    # remove single char
    text = re.sub(r"\b[a-zA-Z]\b", "", text)

    #Remove repeating char
    text = re.sub(r'(\w)\1+\b', r'\1',text)

    #Remove repeating word
    text = re.sub(r'\b(\w+)(\1)+\b', r'\1',text)

    # Tokenisasi
    return word_tokenize(text)

def stopwords_removal(words):
    # Get stopword bahasa Indonesia dari NLTK
    list_stopwords = stopwords.words('indonesian')

    # Baca file stopwords dari file teks
    txt_stopword = pd.read_csv("predic/stopwordsID.txt", header=None)

    # Konversi stopwords dari teks ke list dan tambahkan ke list stopwords
    list_stopwords.extend([x for x in txt_stopword[0]])

    # Konversi list stopwords menjadi set
    list_stopwords = set(list_stopwords)

    # Return hasil preprocessing teks
    filtered_words = [word for word in words if word not in list_stopwords]

    return filtered_words

def build_normalizad_word_dict():
    # Baca file kamus singkatan
    normalizad_word = pd.read_excel("predic/kamus_singkatan.xlsx")
    normalizad_word_dict = {}

    # Buat kamus dari data kamus singkatan
    for index, row in normalizad_word.iterrows():
        if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1]

    return normalizad_word_dict
    
def normalized_term(document, normalizad_word_dict):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]       

def predict(string_data):
    # Baca file .pkl
    with open('predic/transformer.pkl', 'rb') as file:
       transformer = pickle.load(file)

    # Muat model Logistic Regression
    with open('predic/Logistic Regression.pkl', 'rb') as f:
        logreg_model = pickle.load(f)

    # Muat model Decision Tree
    with open('predic/Decision Tree.pkl', 'rb') as f:
        dt_model = pickle.load(f)

    # Muat model Random Forest
    with open('predic/Random Forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    # Muat model K-Nearest Neighbors
    with open('predic/K-Nearest Neighbors.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    # Muat model Gradient Boosting
    with open('predic/Gradient Boosting.pkl', 'rb') as f:
        gb_model = pickle.load(f)

    # Muat model Naive Bayes
    with open('predic/Multinomial Naive Bayes.pkl', 'rb') as f:
        nb_model = pickle.load(f)

    # Muat model true label
    with open('predic/true_labels.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # transtorm string_data menggunakan pre-trained transformer 
    numerical_input = transformer.transform([string_data])
    
    # list model yang digunakan
    models_to_use = [logreg_model, dt_model, rf_model, knn_model, gb_model, nb_model]
    names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors', 'Gradient Boosting', 'Multinomial Naive Bayes']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    #dictionary untuk menyimpan hasil
    predictions = []
    model_names = []
    confidences = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    total_bullying_confidence = 0.0
    total_non_bullying_confidence = 0.0

    for model, model_name in zip(models_to_use, names):
        prediction = model.predict(numerical_input)

        #digunakan untuk memperoleh probabilitas prediksi dari model.
        prediction_proba = model.predict_proba(numerical_input)
        confidence = prediction_proba.max()

        predictions.append(prediction[0])
        model_names.append(model_name)
        confidences.append(confidence)

        # Menjumlahkan nilai confidence untuk bullying dan non-bullying
        if prediction[0] == 'Bullying':
            total_bullying_confidence += confidence
        else:
            total_non_bullying_confidence += confidence

        # memberikan Kesimpulan
        if total_bullying_confidence > total_non_bullying_confidence:
            predicted_class = str(prediction[0])  # Convert to string
            tokenized_data = word_tokenize(string_data)
            conclusion = "karena Total Bullying Score lebih besar dari Total Non-Bullying Score, maka kalimat cenderung dikategorikan sebagai Bullying"
        else:
            predicted_class = str(prediction[0])  # Convert to string
            tokenized_data = word_tokenize(string_data)
            conclusion = "karena Total Bullying Score lebih kecil dari Total Non-Bullying Score, maka Kalimat cenderung dikategorikan sebagai Non-bullying"

        #generate performance model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='Bullying')
        recall = recall_score(y_test, y_pred, pos_label='Bullying')
        f1_ = f1_score(y_test, y_pred, pos_label='Bullying')

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_)


    #--------------------Visualisasi-------------------#
    # Membuat folder static jika belum ada
    if not os.path.exists('static'):
        os.makedirs('static')

    # Generate word cloud
    wc = wordcloud.WordCloud(collocations=False, colormap="tab20b").generate(" ".join(tokenized_data))
    
    plt.figure(figsize=(9, 5))
    plt.title("Kata hasil preprocessing ")
    plt.imshow(wc, interpolation='antialiased')
    _ = plt.axis("off")

    # Simpan WordCloud ke dalam file temporary
    tmp_filepath3 = 'static/wordcloud.png'
    plt.savefig(tmp_filepath3)

    # Plot line matrik
    plt.figure(figsize=(8.5, 4))
    for model, model_name, acc, prec, rec, f1 in zip(models_to_use, names, accuracy_scores, precision_scores, recall_scores, f1_scores):
        plt.plot(metrics, [acc, prec, rec, f1], marker='o', linestyle='-', label=model_name)

    plt.title('Performance Metrics for Machine Learning Models')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.legend(loc='lower right', fontsize='small')  

    # Simpan grafik ke dalam file temporary
    tmp_filepath = 'static/performance_metrics.png'
    plt.savefig(tmp_filepath)

    # Plot pie chart
    # Hitung jumlah prediksi berdasarkan kategori
    category_counts = {
        'Bullying': total_bullying_confidence,
        'Non-Bullying': total_non_bullying_confidence
    }
    
    # Buat list kategori dan jumlah prediksi
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    plt.figure(figsize=(4, 4))
    plt.pie(counts, labels=categories, autopct='%1.1f%%')
    plt.title('Prediction Distribution')

    # Simpan grafik ke dalam file temporary
    tmp_filepath2 = 'static/Prediction_Distribution.png'
    plt.savefig(tmp_filepath2)

    return predictions, model_names, confidences, total_bullying_confidence, total_non_bullying_confidence, conclusion, accuracy_scores, precision_scores, recall_scores, f1_scores, tmp_filepath, tmp_filepath2, tmp_filepath3

def stem_text(text):
    # Membuat objek Stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Melakukan stemming
    stemmed_sentence = stemmer.stem(text)

    # Mengembalikan teks hasil stemming
    return stemmed_sentence


if __name__ == '__main__':
    app.debug = True
    app.run()