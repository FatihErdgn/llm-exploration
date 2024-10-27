# utils/data_processing.py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
from config import MAX_LEN, VOCAB_SIZE
import os
import numpy as np

# Hugging Face veri setini yükle
def load_data():
    os.environ["HF_TOKEN"] = "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"
    data = load_dataset("nvidia/HelpSteer", token=os.getenv("HF_TOKEN"))
    return data

# Tokenizer işlemleri ve veri hazırlığı
def preprocess_data(data):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(data['train']['prompt'])
    
    prompts = tokenizer.texts_to_sequences(data['train']['prompt'])
    responses = tokenizer.texts_to_sequences(data['train']['response'])
    
    # Her iki diziyi de MAX_LEN ile pad edin
    prompts = pad_sequences(prompts, maxlen=MAX_LEN, padding='post')
    responses = pad_sequences(responses, maxlen=MAX_LEN, padding='post')
    
    # decoder_input_data ve decoder_target_data hazırlayın
    decoder_input_data = np.array([response[:-1] for response in responses])
    decoder_target_data = np.array([response[1:] for response in responses])
    
    # Pad edilmesini sağla
    decoder_input_data = pad_sequences(decoder_input_data, maxlen=MAX_LEN, padding='post')
    decoder_target_data = pad_sequences(decoder_target_data, maxlen=MAX_LEN, padding='post')
    
    return prompts, decoder_input_data, decoder_target_data, tokenizer

def save_tokenizer(tokenizer, path='models/tokenizer.pickle'):
    import pickle
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(path='models/tokenizer.pickle'):
    import pickle
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
