from flask import Flask, request
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

@app.route('/calculate', methods=['POST'])
def calculate():
    x1, x2 = request.json['x1'], request.json['x2']
    operator = request.json['operator']
    model = None
    if operator == '+':
        model = load_model(os.getcwd()+'\\machine_learning\\models\\addition_model')
    elif operator == '-':
        model = load_model(os.getcwd()+'\\machine_learning\\models\\subtraction_model')
    else:
        model = None

    prediction = model.predict(np.array([[x1, x2]]))[0][0]

    return {
        "x1": x1,
        "x2": x2,
        "operator": operator,
        "prediction": float(prediction)
    }

if __name__ == "__main__":
    app.run(debug=True)