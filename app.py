import email
import re
from flask import Flask, render_template,request
import pickle 
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
	if request.method=='POST':
		features=[
		 request.form['texture_mean'],
		 request.form['area_mean'],
		 request.form['smoothness_mean'],
		 request.form['concavity_mean'],
		 request.form['symmetry_mean'],
		 request.form['fractal_dimension_mean'],
		 request.form['texture_se'],
		 request.form['area_se'],
		 request.form['smoothness_se'],
		 request.form['concavity_se'],
		 request.form['symmetry_se'],
		 request.form['fractal_dimension_se'],
		 request.form['smoothness_worst'],
		 request.form['concavity_worst'],
		 request.form['symmetry_worst'],
		 request.form['fractal_dimension_worst']]

		
		lst=map(lambda x: float(x), features)
		values=np.array(list(lst))
		values=values.reshape(1,-1)

		pred=model.predict(values)

	return render_template('result.html',pred=pred)
	
	return render_template('index.html')

if __name__=='main':
	app.run(debug=True)
