# app.py
from flask import Flask, render_template, request
import numpy as np
from utils.data_processing import load_tokenizer
from utils.model_definition import define_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_LEN

app = Flask(__name__)

# Model ve tokenizer'ı yükle
model = define_model()
model.load_weights('models/gpt_model.h5')
tokenizer = load_tokenizer()

# Yanıt üretme fonksiyonu (daha önce tanımlanan decode_sequence fonksiyonu ile benzer)
def decode_sequence(input_seq):
    states_value = model.layers[3].predict(input_seq)  # Encoder'dan durumu al
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_sentence = ""
    
    while not stop_condition:
        output_tokens, h, c = model.layers[5].predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')
        decoded_sentence += ' ' + sampled_word

        if (sampled_word == 'end' or len(decoded_sentence.split()) > MAX_LEN):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
        
    return decoded_sentence

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Yanıt üretme işlevi
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input_sequence, maxlen=MAX_LEN, padding='post')
    response = decode_sequence(input_sequence)
    return render_template('index.html', input_text=input_text, response=response)

if __name__ == '__main__':
    app.run(debug=True)
