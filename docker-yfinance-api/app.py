from flask import Flask, request, jsonify
from lstm import lstm
import os


# Load model

# Initialize a Flask app
app = Flask(__name__)

# Create an API endpoint
@app.route('/')
def hello_world():
    return 'Hello World!!!'


@app.route('/api/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        query_parameters = request.args
        if query_parameters:
            symbol = query_parameters.get('symbol')
            result = {
                'symbol': symbol,
                'predict': lstm(symbol)
            }
            return jsonify(result)
        else:
            return 'Select Symbol like /api/predict?symbol=AMD'
    else:
        return 'Do GET action!!'

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host ='0.0.0.0', port=port, debug=True)