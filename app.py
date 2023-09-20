from flask import Flask, render_template, request, redirect, url_for
from ml import *

app = Flask(__name__)

sns.set()
warnings.filterwarnings('ignore')


@app.route('/')
def home():
  return render_template('Project.html')


@app.route('/predict', methods=['POST'])
def upload_file():
  gender = request.form.get('gender')
  age = request.form.get('age')
  educ = request.form.get('education')
  ses = request.form.get('ses')
  mmse = request.form.get('mmse')
  etiv = request.form.get('etiv')
  nwbv = request.form.get('nwbv')
  asf = request.form.get('asf')
  data = [{
      'M/F': gender,
      'Age': age,
      'EDUC': educ,
      'SES': ses,
      'MMSE': mmse,
      'eTIV': etiv,
      'nWBV': nwbv,
      'ASF': asf
  }]
  X_test = pd.DataFrame(data)

  # Scale input data
  X_test_scaled = scaler.transform(X_test)

  # Make predictions
  y_pred = SelectedRFModel.predict(X_test_scaled)

  # Print predicted labels
  #print(y_pred)
  n = ''
  if (y_pred == 0):
    n = 'NOT AFFECTED WITH DELIRIUM'
  else:
    n = 'AFFECTED WITH DELIRIUM'
  return render_template('Project1.html', n=n)


# Display results

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
