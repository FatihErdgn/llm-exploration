# app.py
from flask import Flask, render_template, request, jsonify
from utils.data_processing import load_tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from config import MAX_LEN

app = Flask(__name__)

# Tokenizer ve modeli yükle
tokenizer = load_tokenizer('models/tokenizer.pickle')
model = load_model('models/gpt_model.h5')  # tf formatında kaydedildiyse

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get("input_text")
    print("Input Text:", input_text)  # Kullanıcıdan gelen giriş metni
    
    # Girdiyi tokenize et ve pad et
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input_sequence, maxlen=MAX_LEN, padding='post')
    print("Tokenized & Padded Input Sequence:", input_sequence)  # Tokenize edilmiş ve pad edilmiş giriş
    
    # Model tahmini yap
    prediction = model.predict([input_sequence, np.zeros_like(input_sequence)])  # Örnek decoder girişi için sıfır
    print("Prediction Output:", prediction)  # Modelin tahmini (tensör verisi)
    
    predicted_sequence = np.argmax(prediction, axis=-1)
    print("Predicted Sequence (Token IDs):", predicted_sequence)  # Tahmin edilen token ID'leri

    # Yanıtı çöz ve döndür
    decoded_words = [tokenizer.index_word.get(idx, "") for idx in predicted_sequence[0] if idx != 0]
    print("Decoded Words:", decoded_words)  # Token ID'lerinin kelimelere dönüştürülmüş hali
    
    response_text = " ".join(decoded_words)
    print("Response Text:", response_text)  # Nihai yanıt metni

    # Tahmini index.html'e gönder
    return render_template('index.html', input_text=input_text, response=response_text)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)