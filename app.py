from flask import Flask, render_template, redirect
from flask import request
import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from importing import TextSelector,NumberSelector
import joblib

def salary_category(bin):

    if bin==1:
        return "Job salary should be between $50000 and $75000"
    if bin==2:
        return "Job salary should be between $75000 and $100000"
    if bin==3:
        return "Job salary should be between $100000 and $125000"
    if bin==4:
        return "Job salary should be between $125000 and $150000"
    if bin==5:
        return "Job salary should be between $150000 and $175000"
    if bin==6:
        return "Job salary should be between $175000 and $200000"
    if bin==7:
        return "Job salary should be more than $200000"

def skill_check(dict):
    v=dict.values()
    skill_dict={
    'Python':0, 'R':0, 'SQL':0, 'AWS':0, 'Excel':0, 'GCP':0, 'Azure':0, 'Spark':0,
       'PyTorch':0, 'TensorFlow':0, 'Tableau':0, 'Keras':0, 'NoSQL':0, 'Scikit-Learn':0,
       'Machine_Learning':0, 'Hadoop':0, 'Scala':0, 'Data_Brick':0
    }
    for i in v:
        if (i in skill_dict.keys()):
            skill_dict[i]=1
    return skill_dict

model = joblib.load('salary_predict_model.pkl')


class TextSelector(BaseEstimator, TransformerMixin):

        def __init__(self, key):
            self.key = key

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    
        def __init__(self, key):
            self.key = key

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[[self.key]]
app = Flask(__name__)

@app.route("/")
def index():
    
    listings = {'_id':'615f3911a6d569a7cd86c945',
 'url': 'https://www.cnbc.com/2021/10/04/here-are-the-five-most-valuable-college-majors-.html',
 'img': 'https://image.cnbcfm.com/api/v1/image/106885838-1621509859148-gettyimages-1318999756-vcg111330726320.jpeg?v=1633354738&w=929&h=523',
 'title': 'Here are the 5 most valuable — and 5 least valuable — college majors'}
    return render_template("index.html",listings=listings)

@app.route("/slides")
def slides():
    return render_template("slides.html")

@app.route("/job_cat")
def jm():
    return render_template("jobs_by_category_mw.html")
    

@app.route("/job_market")
def jc():
    return render_template("job_market_map_visual_mw.html")
    

@app.route("/salary")
def sa():
    return render_template("as_charts.html")

@app.route("/employment_info")
def ei():
    return render_template("as_map.html")

@app.route('/salary_prediction',methods=['GET', 'POST'])
def sp():
    if request.method == 'POST':
        names = request.form.to_dict()
        skill=skill_check(names)
        input=[names["titles"],names["size"],names["ownership"],names["industry"],names["sector"],names["state"],names["age"]]
        input=input+list(skill.values())
        input=input+[names["titles"],names["seniority"]]
        df=pd.DataFrame([input],columns=['Job Title', 'Size', 'Type of ownership', 'Industry', 'Sector', 'State',
       'Age', 'Python', 'R', 'SQL', 'AWS', 'Excel', 'GCP', 'Azure', 'Spark',
       'PyTorch', 'TensorFlow', 'Tableau', 'Keras', 'NoSQL', 'Scikit-Learn',
       'Machine_Learning', 'Hadoop', 'Scala', 'Data_Brick', 'Job',
       'Seniority'])
        pred=salary_category( model.predict(df)[0])
        return render_template('salary_prediction_sj.html',pred=pred)
    else:
        return render_template('salary_prediction_sj.html')


if __name__ == "__main__":
    from imports import TextSelector,NumberSelector
    app.run(debug=True)
