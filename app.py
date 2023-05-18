import joblib
import numpy as np
import pandas as pd
import io     # added import statement
from flask import Flask, request, jsonify   # added import statement

app = Flask(__name__)

MODEL_PATH = "models/best_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route('/predict',methods=['POST'])
def predict():
    content_type = request.content_type
    if content_type == 'application/json':
        data = request.get_json()
        x_test = np.array(data["inputs"]).reshape(1,-1)
    elif content_type == 'text/csv':
        data = request.data.decode('utf-8')
        x_test = pd.read_csv(io.StringIO(data)).values
    else:
        return jsonify(Error="Unsupported media type"), 415
    
    y_pred = model.predict(x_test)
    prediction = str(int(y_pred[0]))

    return jsonify(prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
