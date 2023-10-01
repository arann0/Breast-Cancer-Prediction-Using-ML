from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained SVM model
model = joblib.load('breast_cancer_prediction_model.sav')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        result = 'This Person have Breast Cancer' if prediction[0] == 1 else 'The Person does not have Breast Cancer'
        return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)