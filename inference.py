import pickle
import numpy as np
import pandas as pd
import os
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Question 2
# load model
model = pickle.load(open(r'churn_model.pkl', 'rb'))


def number_convert(number):
    """
    convert number from string to int or float(depends if number has a '.')
    :param number: str
    :return: int/float
    """
    if number.isdigit():
        return int(number)
    else:
        return float(number)


@app.route('/predict_churn')
def predict_churn():
    is_male = number_convert(request.args.get('is_male'))
    num_inters = number_convert(request.args.get('num_inters'))
    late_on_payment = number_convert(request.args.get('late_on_payment'))
    age = number_convert(request.args.get('age'))
    years_in_contract = number_convert(request.args.get('years_in_contract'))
    single_sample = np.array([is_male, num_inters, late_on_payment, age, years_in_contract]).reshape(1, 5)
    sample_pred = model.predict(single_sample)
    return f'{sample_pred[0]}'


@app.route('/predict_churn_bulk', methods=['POST'])
def predict_churn_bulk():
    new_quark = json.loads(request.get_json())
    query = pd.DataFrame(new_quark)
    prediction = model.predict(query)
    prediction = list(map(int, prediction))
    return jsonify({'prediction': prediction})


# def model_check(model_input):
#     """
#     check if model predictions are the same as the training predictions.
#     :param model_input: model
#     :return: None
#     """
#     X_test = pd.read_csv(r'X_test.csv')
#     preds = np.loadtxt(r'preds.csv')
#     y_pred = model_input.predict(X_test)
#     print(np.array_equal(y_pred, preds))
#
#
# def main():
#     # Question 3
#     # model_check(model)


if __name__ == '__main__':
    # main()
    port = os.environ.get('PORT')
    app.run(host='0.0.0.0', port=int(port))
