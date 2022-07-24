import numpy as np
from flask import render_template, request
from flask import Flask
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

model_name = "bert-base-nli-mean-tokens"
model = SentenceTransformer(model_name)

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit():
    sentences = []
    textone = request.form['textone']
    texttwo = request.form['texttwo']
    sentences.append(textone)
    sentences.append(texttwo)
    sentence_vecs = model.encode(sentences)
    score = cosine_similarity([sentence_vecs[0]],sentence_vecs[1:])[0][0]
    score = np.round(score, 1)
    score2 = score*100
    return render_template("home.html", prediction=score, bar_prediction=score2)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)