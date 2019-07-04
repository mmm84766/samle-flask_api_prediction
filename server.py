# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import requests

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/api', methods=['GET','POST'])

def predict():
 if request.method=='GET':
    url = 'http://localhost:2000/api'
    r = requests.post(url,json={'exp':1.8,})
    return("1.8 years of experience results in a salary of: " + str(r.json()) + " USD")
 elif request.method=='POST':
    data = request.get_json(force=True)
    print(data)
 # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])
 # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)
 else:
    return("ELSE")



if __name__ == '__main__':
    app.run(port=2000, debug=True)
