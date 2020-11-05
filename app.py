from flask import Flask, render_template, request
import pickle
import numpy as np
model = pickle.load(open('music.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('genre.html')

@app.route('/predict', methods = ['POST'])
def genre():
    data1 = request.form['a']
    data2 = request.form['b']
    a = np.array([[data1,data2]])
    p = model.predict(a)
    return render_template('predict.html', data=p)
if __name__ == "__main__":
    app.run(debug=True)

