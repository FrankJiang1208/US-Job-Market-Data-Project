# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
import joblib

stopWords = set(stopwords.words('english'))


# %%
df = pd.read_csv('static/resources/data.csv')


# %%
df.head()


# %%
#Get the datas with salary with glassdoor estimate
df['Salary_est'] = df['Salary Estimate'].apply(lambda x: 1 if 'glassdoor est.' in x.lower() else 0)
df = df[df['Salary_est'] == 1]


# %%
# Calculate average salary and drop non-useful columns
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

df['min_salary'] = minus_Kd.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = minus_Kd.apply(lambda x: int(x.split('-')[1]))
df['Salary'] = (df.min_salary+df.max_salary)*1000/2
df = df.drop(['Salary Estimate', 'Salary_est', 'min_salary', 'max_salary'],1)


# %%
# Company name text only
df['Company Name'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'].split('\n')[0], axis = 1)


# %%
# Create new features such as city & state 
df['City'] = df['Location'].apply(lambda x: x.split(',')[0] if ',' in x.lower() else x)
df['State'] = df['Location'].apply(lambda x: x.split(',')[1] if ',' in x.lower() else x) 
df = df.drop(['Location'],1)
df.head()


# %%
# Calculate the age of company 
df['Founded'] = df['Founded'].apply(lambda x: np.nan if x in ['Company - Private', '-1', 'Company - Public', 'Unknown', 'Contract', 'Nonprofit Organization', 'Self-employed'] else x)
df['Age'] = df['Founded'].apply(lambda x: x if x is np.nan else 2021 - int(x))
df = df.drop(['Founded'],1)


# %%
# Python
df['Python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
 
# R studio 
df['R'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)

# SQL 
df['SQL'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)

# AWS 
df['AWS'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

# Excel
df['Excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

# Google Cloud
df['GCP'] = df['Job Description'].apply(lambda x: 1 if 'google cloud' in x.lower() or 'gcp' in x.lower() else 0)

# Microsoft Azure
df['Azure'] = df['Job Description'].apply(lambda x: 1 if 'microsoft azure' in x.lower() or 'azure' in x.lower() else 0)

# Spark
df['Spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

# PyTorch
df['PyTorch'] = df['Job Description'].apply(lambda x: 1 if 'pytorch' in x.lower() else 0)

# TensorFlow
df['TensorFlow'] = df['Job Description'].apply(lambda x: 1 if 'tensorflow' in x.lower() else 0)

# Tableau
df['Tableau'] = df['Job Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)

# Keras
df['Keras'] = df['Job Description'].apply(lambda x: 1 if 'keras' in x.lower() else 0)

#NoSQL
df['NoSQL'] = df['Job Description'].apply(lambda x: 1 if 'nosql' in x.lower() else 0)

#Sci-kit Learn
df['Scikit-Learn'] = df['Job Description'].apply(lambda x: 1 if 'scikit learn' in x.lower() else 0)

#Machine Learning
df['Machine_Learning'] = df['Job Description'].apply(lambda x: 1 if 'machine learning' in x.lower() else 0)

#Hadoop
df['Hadoop'] = df['Job Description'].apply(lambda x: 1 if 'hadoop' in x.lower() else 0)

# Scala
df['Scala'] = df['Job Description'].apply(lambda x: 1 if 'scala' in x.lower() else 0)

# Data Brick
df['Data_Brick'] = df['Job Description'].apply(lambda x: 1 if 'data brick' in x.lower() else 0)


# %%
# Convert size to lower case
df['Size'] = df['Size'].str.lower()
df.head()


# %%
df['Job Title'] = df['Job Title'].apply(lambda x: ' '.join(x.split()))
df['Job Title'] = df['Job Title'].apply(lambda x: re.sub(r'[^\w\s]',' ', x.lower()))
df['Job Description'] = df['Job Description'].apply(lambda x: ' '.join(x.split()))
df['Job Description'] = df['Job Description'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
df['Job Description'] = df['Job Description'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopWords]))   


# %%
# Classifying job titles into each category
def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'data analyst'
    elif 'machine learning' in title.lower():
        return 'machine learning engineer'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'data science related jobs'


# %%
# Classifying job levels into each category
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'vp' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower()or 'experienced' in title.lower() or 'iii' in title.lower() or 'research' in title.lower() or 'manager' in title.lower():
        return 'Senior'
    elif 'intermediate' in title.lower() or 'staff' in title.lower() or 'ii' in title.lower():
        return 'Mid'
    elif 'jr' in title.lower() or 'junior' in title.lower() or 'intern' in title.lower() or 'student' in title.lower()or 'associate' in title.lower():
        return 'Junior'
    else:
        return 'Not Specified'


# %%
# Process data based on above defined functions 
df['Job'] = df['Job Title'].apply(title_simplifier)
df['Seniority'] = df['Job Title'].apply(seniority)


# %%
# Categorize salary
def salary_category(salary):

    if 50000 <= salary < 75000:
        return 1
    if 75000 <= salary < 100000:
        return 2
    if 100000 <= salary < 125000:
        return 3
    if 125000 <= salary < 150000:
        return 4
    if 150000 <= salary < 175000:
        return 5
    if 175000 <= salary < 200000:
        return 6
    else:
        return 7
target = df['Salary'].apply(salary_category)


# %%
# Fill all NaN values in numeric & text features 
df['Age'] = df['Age'].fillna(0)
df = df.fillna('Unknown')


# %%
class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


# %%
df=df.drop(["Job Description","Rating","Company Name","City","Revenue"],axis=1)


# %%
features= [c for c in df.columns.values if c not in ['Salary']]
numeric_features= [c for c in df.columns.values if c not in ['Job Title','Size','Type of ownership',
                                                            'Industry','Sector','State','Job','Seniority']]

X_train, X_test, y_train, y_test = train_test_split(df[features], target, test_size=0.2, random_state=42)
X_train.head(2)


# %%

# Define text feature pipelines
jt=TextSelector(key='Job Title')
s=TextSelector(key='Size')
o=TextSelector(key='Type of ownership')
i=TextSelector(key='Industry')
sec=TextSelector(key='Sector')
state= TextSelector(key='State')
j=TextSelector(key='Job')
Job_Title = Pipeline([
                ('selector', jt),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            ])

# Job_Desc = Pipeline([
#                 ('selector', TextSelector(key='Job Description')),
#                 ('tfidf', TfidfVectorizer(stop_words='english',ngram_range=(1,2), max_features=1000))
#             ])

# Company = Pipeline([
#                 ('selector', TextSelector(key='Company Name')),
#                 ('tfidf', TfidfVectorizer(stop_words='english'))
#             ])

Size = Pipeline([
                ('selector', s),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            ])

Ownership = Pipeline([
                ('selector', o),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            ])

Industry = Pipeline([
                ('selector', i),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            ])

Sector = Pipeline([
                ('selector', sec),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            ])

# Revenue = Pipeline([
#                 ('selector', TextSelector(key='Revenue')),
#                 ('tfidf', TfidfVectorizer(stop_words='english'))
#             ])

# City = Pipeline([
#                 ('selector', TextSelector(key='City')),
#                 ('tfidf', TfidfVectorizer(stop_words='english'))
#             ])

State = Pipeline([
                ('selector',state),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            ])

Job = Pipeline([
                ('selector',j ),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            ])

Job_Title.fit_transform(X_train)


# %%
# Define numeric feature pipelines 
# Rating =  Pipeline([
#                 ('selector', NumberSelector(key='Rating')),
#                 ('standard', StandardScaler())
#             ])
a=NumberSelector(key='Age')
p=NumberSelector(key='Python')
r= NumberSelector(key='R')
sql=NumberSelector(key='SQL')
aws=NumberSelector(key='AWS')
excel=NumberSelector(key='Excel')
gcp=NumberSelector(key='GCP')
az=NumberSelector(key='Azure')
spark=NumberSelector(key='Spark')
t=NumberSelector(key='PyTorch')
tensor=NumberSelector(key='TensorFlow')
tab=NumberSelector(key='Tableau')
k= NumberSelector(key='Keras')
nosql= NumberSelector(key='NoSQL')
sk=NumberSelector(key='Scikit-Learn')
ml= NumberSelector(key='Machine_Learning')
h=NumberSelector(key='Hadoop')
scala=NumberSelector(key='Scala')
db=NumberSelector(key='Data_Brick')
Age =  Pipeline([
                ('selector', a),
                ('standard', StandardScaler())
            ])
Python =  Pipeline([
                ('selector', p),
                ('standard', StandardScaler())
            ])
R =  Pipeline([
                ('selector',r),
                ('standard', StandardScaler()),
            ])
SQL =  Pipeline([
                ('selector', sql),
                ('standard', StandardScaler()),
            ])
AWS =  Pipeline([
                ('selector', aws),
                ('standard', StandardScaler()),
            ])
Excel =  Pipeline([
                ('selector', excel),
                ('standard', StandardScaler()),
            ])
GCP =  Pipeline([
                ('selector', gcp),
                ('standard', StandardScaler()),
            ])
Azure =  Pipeline([
                ('selector', az),
                ('standard', StandardScaler()),
            ])
Spark =  Pipeline([
                ('selector', spark),
                ('standard', StandardScaler()),
            ])
PyTorch =  Pipeline([
                ('selector', t),
                ('standard', StandardScaler()),
            ])
TensorFlow =  Pipeline([
                ('selector', tensor),
                ('standard', StandardScaler()),
            ])

Tableau =  Pipeline([
                ('selector', tab),
                ('standard', StandardScaler()),
            ])

Keras =  Pipeline([
                ('selector',k),
                ('standard', StandardScaler()),
            ])
NoSQL =  Pipeline([
                ('selector',nosql),
                ('standard', StandardScaler()),
            ])
Scikit_Learn =  Pipeline([
                ('selector', sk),
                ('standard', StandardScaler()),
            ])
Machine_Learning =  Pipeline([
                ('selector', ml),
                ('standard', StandardScaler()),
            ])
Hadoop =  Pipeline([
                ('selector', h),
                ('standard', StandardScaler()),
            ])
Scala =  Pipeline([
                ('selector', scala),
                ('standard', StandardScaler()),
            ])
Data_Brick =  Pipeline([
                ('selector', db),
                ('standard', StandardScaler()),
            ])


# %%
# FeatureUnion for all feature vectors 
feats = FeatureUnion([('Job Title', Job_Title), 
                    #   ('Job Description', Job_Desc),
                    #   ('Company Name', Company),
                      ('Company Size', Size),
                      ('Type of ownership', Ownership),
                      ('Industry', Industry),
                      ('Sector', Sector),
                    #   ('Revenue', Revenue),
                      # ('City', City),
                      ('State', State),
                      ('Job', Job),
                      # ('Rating', Rating),
                      ('Age', Age),
                      ('Python', Python),
                      ('R', R),
                      ('SQL', SQL),
                      ('AWS', AWS),
                      ('Excel', Excel),
                      ('GCP', GCP),
                      ('Azure', Azure),
                      ('Spark', Spark),
                      ('PyTorch', PyTorch),
                      ('TensorFlow', TensorFlow),
                      ('Tableau', Tableau),
                      ('NoSQL', NoSQL),
                      ('Scikit_Learn', Scikit_Learn),
                      ('Machine_Learning',Machine_Learning),
                      ('Hadoop', Hadoop),
                      ('Scala',Scala),
                      ('Data_Brick',Data_Brick)
                     ])

feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(X_train)


# %%
pipeline = Pipeline([
    ('features',feats),
    ('classifier', RandomForestClassifier(random_state = 42)),
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
preds


# %%
# Define a custom accuracy evaluation function 
def evaluate(preds, y_test, variation):
    counter = 0
    for i in range(len(preds)):
        if y_test[i]-variation <= preds[i] <= y_test[i]+variation:
            counter += 1
        else: 
            counter += 0
    accuracy = counter / len(preds) 
    return accuracy
        
evaluate(list(preds), list(y_test), 1)


# %%
joblib.dump(pipeline, 'salary_predict_model.pkl')


# %%
l=[x**2 for x in range(36)]
l[1:10:3]


# %%



