import os, sys, time, json, keras, numpy
from keras.preprocessing import sequence
from keras.models import model_from_json
from numpy import array
from flask import Flask, request, jsonify, flash
import json
import tensorflow as tf

app = Flask(__name__)
max_review_length = 500
numpy.random.seed(42)
top_words = 100000

NUM_WORDS=10000
INDEX_FROM=3
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}

root = os.path.abspath(".")
model_dir = root + "/models/"

def load_model():
    json_file = open(model_dir + "deepsentiment_keras.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_dir + "deepsentiment_keras.h5")
    print("Loaded model from disk")
    global graph
    graph = tf.get_default_graph()

    return loaded_model

model= load_model()

def predict_sentiment(m, review):
    tmp = []
    for word in review.lower().split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length)
    pred = m.predict(array([tmp_padded][0]))[0][0]
    return pred

@app.route('/sentiment/v1/predict', methods=["POST"])
def predict ():
    data = {"success": False}
    text= request.json["text"]


    with graph.as_default():
        res = predict_sentiment(model,text)
    print(res)
    data["success"] = True
    data["prediction"] = str(res)
    return jsonify(data)

if __name__ == "__main__":
    print("* Starting web service...")
    app.run(
        host='0.0.0.0',
        port=3004,
        debug="true"
    )

