#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import joblib
import os
import json


# In[3]:


from functions import extract_file, preprocess, predict_roles


# In[ ]:


app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# In[ ]:


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(open(os.path.join(BASE_DIR, 'nbmodel.pkl'), 'rb'))
vectorizer = joblib.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))
skills_data = json.load(open(os.path.join(BASE_DIR, 'skills.json')))
salary_data = json.load(open(os.path.join(BASE_DIR, 'job_salaries.json')))


# In[ ]:


@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return jsonify({'error':'no file part'})
    file = request.files['resume']
    if file.filename == "":
        return jsonify({'error':'no file selected'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    file.save(filepath)
    
    text=extract_file(filepath)
    cleaned = preprocess(text)
    result = predict_roles(model,vectorizer,cleaned)
    
    return render_template('index.html', roles= result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0', port = port)


# In[ ]:




