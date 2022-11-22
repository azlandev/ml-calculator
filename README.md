# ML-Calculator

A calculator that uses machine learning to predict answers. This application uses TensorFlow and Keras for machine learning. The front-end is a simple React app that sends a query to a Flask server through the server's API. 

## Installation

After downloading everything in this repository, Anaconda can be used to install the requirements in a virtual environment.

```
conda create --name <env> --file requirements.txt
```
To install the node packages, run `npm install` in the `/client/` folder.

Note: Installing the requirements from `requirements.txt` installs `tensorflow-gpu`. If you do not have a TensorFlow compatiple GPU, you should install the non-GPU version of TensorFlow instead. Please refer to TensorFlow's documentation for more installation details. 

## Usage

To start the flask server, simply run the `flask_server.py` by entering `python flask_server.py` in your command line. To start the React app, enter `npm start` in a separate command line window from the `/client/` folder. A new window should automatically open in your browser 

## Machine learning

The machine learning models were created using the Keras API on top of TensorFlow. The pre-trained models for all operations are stored in `/machine-learning/models/` in the [SavedModel](https://www.tensorflow.org/guide/saved_model) format. You are free to create your own models for this application. Two scripts are included in the `/machine-learning/` folder to generate data and create/train a neural network. 

The models for all four operations (+,-,*,/) use linear regression to calculate weights. It's simply a sequential model with a normalization layer and a single Dense layer. 
```
add_model = tf.keras.models.Sequential([
    normalizer,
    layers.Dense(units=1)
])
```
Since addition and subtraction are relatively simple for the neural network to learn, the normalization layer uses built in preprocessing functions to adapt to the training data.
```
normalizer = preprocessing.Normalization()
normalizer.adapt(train_data)
```
Multiplication and division on the other hand are log normalized so that a linear regression technique can be applied to them.
```
train_normalized = np.log(train_data)
test_normalized = np.log(test_data)
target_normalized = np.log(div_target)
test_target_normalized = np.log(test_div_target)
normalizer = preprocessing.Normalization()
normalizer.adapt(train_normalized)
```
This comes with the trade-off of not accepting inputs that are <= 0 and requiring the predicted output to be raised to the power of e.

## Flask server

The back-end of this application uses Flask. The server uses an API that accepts a POST request containing two floats, `x1` and `x2`, as well as a string `operator`. If the `operator` string is either "+" or "-", `x1` and `x2` are simply passed to the prediction function. If the `operator` string is either "×" or "÷", the natural log of `x1` and `x2` are passed to the prediction function and the returned value is raised to the power of e due to the multiplication and division models being trained on log normalized data.

## React front-end

The calculator component is located in `/client/src/components/Calculator.js`. Query data is sent to the server by performing a fetch on the API.
```
fetch('/calculate', requestOptions).then(res => res.json()).then(
    data => {dispatch({ 
        type: ACTIONS.EVALUATE, 
        payload: {
            prediction: data.prediction,
            accuracy: 100 - (Math.abs(data.prediction - actualAnswer)/Math.max(Math.abs(data.prediction), Math.abs(actualAnswer)))
        }
    })
    setLoading(false);
    }
)
```
When running the app locally, the server url must be included in `package.json` as a proxy.
```
...
"proxy": "http://127.0.0.1:5000/",
...
```
![Alt text](client/public/example.png?raw=true)
