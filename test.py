# test.py
from utils.data_processing import load_tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_LEN
import numpy as np
# Tokenizer ve modeli yükle
tokenizer = load_tokenizer('models/tokenizer.pickle')
model = load_model('models/gpt_model.h5')

# 5. Modeli Basit Bir Girdi ile Test Et
def test_model(test_text):
    # Girişi hazırlayın
    input_sequence = tokenizer.texts_to_sequences([test_text])
    input_sequence = pad_sequences(input_sequence, maxlen=MAX_LEN, padding='post')
    print("Tokenized Test Input:", input_sequence)

    # Tahmin döngüsü
    predicted_sequence = []
    current_input = np.zeros((1, MAX_LEN))
    for i in range(MAX_LEN):
        prediction = model.predict([input_sequence, current_input])
        predicted_id = np.argmax(prediction[0, i, :])
        
        # Durdurma koşulu: Eğer model sürekli sıfır tahmin ediyorsa döngüyü kır
        if predicted_id == 0:
            break
        
        # Tahmini sonraki giriş olarak kullan
        predicted_sequence.append(predicted_id)
        current_input[0, i] = predicted_id  # Tahmini `decoder` girdisine ekleyin
    
    # Kelimeleri çözümleyin
    decoded_words = [tokenizer.index_word.get(idx, "") for idx in predicted_sequence]
    response_text = " ".join(decoded_words)
    print("Decoded Words:", decoded_words)
    print("Predicted Response:", response_text)

# Test
test_model("What are the key factors for building a helpful assistant?")

