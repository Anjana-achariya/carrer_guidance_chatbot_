#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))


# In[2]:


def preprocess(resume):
    resume = re.sub("[^a-zA-Z]"," ",resume)
    text = resume.lower()
    text = text.split()
    text = [words for words in text if words not in sw]
    text = " ".join(text)
    return text



# In[3]:


import json


# In[4]:


with open("skills.json") as f:
    job_skills = json.load(f)


# In[ ]:


job_skills


# In[6]:


import joblib
model = joblib.load("nbmodel.pkl")
vect = joblib.load("vectorizer.pkl")
encoder = joblib.load("encoder.pkl")


# In[10]:


import fitz
import docx2txt
import string
from docx import Document
import subprocess
# In[12]:


def extract_file(resume):
    fn = resume.filename.lower()
    with open(fn , "wb") as f:
        f.write(resume.read())

    if fn.endswith(".pdf"):
        try:
            pdf_file = fitz.open(fn)
            text = ""
            for page in pdf_file:
                text+=page.get_text()
            return text
        except Exception as e:
            return f"error reading the file {e}"
       

    elif fn.endswith(".docx"):
        try:
            doc = Document(fn)
            text = '\n'.join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            return f"error reading the file {e}"


    elif fn.endswith(".doc"):
        try:
            text = subprocess.check_output(['antiword',fn])
            return text.decode('utf-8')
        except subprocess.CalleddProcessError as e:
            return f"error reading the file {e}"
        except FileNotFoundError:
            return "antiword is not installed on the server. Please install it to process .doc files."

        
    else:
        return "Unsupported file format. Please upload a PDF, DOCX, or DOC file."



# In[13]:


def predict_roles(model,vectorizer,text,top_n=5):
  vec = vect.transform([text])
  probs = model.predict_proba(vec)[0]
  classes = model.classes_
  top_indices = probs.argsort()[-top_n:][::-1]
  top_classes = classes[top_indices]
  top_probs = probs[top_indices]
  return dict(zip(top_classes,top_probs))


# In[ ]:







