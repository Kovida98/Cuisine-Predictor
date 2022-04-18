import ast
import json
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
id=[]
cuisine_type=[]
ingredients_list=[]
ing=[]
ps = PorterStemmer()
def readdata(file):

    f=open(file)
    data=json.load(f)
    #print(data)
    for i in range(len(data)):
        id.append(data[i]["id"])
    for j in range(len(data)):
         cuisine_type.append(data[j]["cuisine"])
    for k in range(len(data)):
         ingredients_list.append(data[k]["ingredients"])
    return id,cuisine_type,ingredients_list

def processingingredients(ingredients_list):

    string=" "
    list=[]
    for i in ingredients_list:
        for j in i:
            x=ps.stem(j)
            y="".join(x.split())
    #        print(y)
            string+=" "+y
        list.append(str(string))
        string=" "
    return list

def processinginput(input,list):

#    input=['paprika','rice krispies','plain flour', 'ground pepper', 'salt', 'tomatoes']
    input_string=""
    input_list=[]
    for k in input:
        s=ps.stem(k)
        input_string+=" "+"".join(s.split())
    input_list.append(input_string)
    list.append(input_string)

    return list


def countvectorizer(list):

    countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
    count_wm = countvectorizer.fit_transform(list)
    vector=count_wm.todense()
    #print(vector)
    return vector


def traindata(vector,cuisine_type):

    ingredientsvector_data=vector[0:len(vector)-1]
    #print(ingredientsvector_data.shape)
    inputvector_data= vector[-1]
    #print(inputvector_data.shape)
    #print(train_data)
    #print(cuisine_type)
    X_train, X_test, y_train, y_test = train_test_split(ingredientsvector_data, cuisine_type, test_size = 0.2,random_state=101)

    return X_train,X_test,y_train,y_test,ingredientsvector_data,inputvector_data

def knnmodel(X_train,X_test,y_train,y_test,ingredientsvector_data,inputvector_data):

    knn=KNeighborsClassifier(n_neighbors=5)
    model1=knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    #print(y_pred)
    test=model1.predict(inputvector_data)
    #print(test)
    #score1=metrics.accuracy_score(y_test, y_pred)
    #print(metrics.accuracy_score(y_test,y_pred))
    score1 = model1.predict_proba(inputvector_data)
    score=max(score1[0])
    #print(score)
    return test,score

def topNcuisines(inputvector_data,ingredientsvector_data,id,N):
    sim = cosine_similarity(inputvector_data,ingredientsvector_data).flatten()

    df1=pd.DataFrame({'score':sim})
    #print(df1)
    df2=pd.DataFrame({'id':id})
    frames=[df1,df2]
    df=pd.concat(frames,axis=1)
    #print(df)
    final_df = df.sort_values(by=["score"],ascending=False)
    #print(final_df)
    #N=5
    result = final_df[["id","score"]].head(int(N))
    #print(result)
    return result

def output_json(res,test,score):
    y=[]
    out= res.to_json(orient = 'table')
    parsed = json.loads(out)
    closest_json=json.dumps(parsed, indent=4)
    di=ast.literal_eval(closest_json)
    del di["schema"]
    for i in di["data"]:
        del i["index"]

    #print(di)
    result_json={}
    result_json["cuisine"]=test[0]
    result_json["score"]=round(score,2)
    result_json["Closest"]=di["data"]
    #print(result_dict)
    output_result=json.dumps(result_json, indent=3)
    #result_dict["Closest"]=out
    print(output_result)
    return result_json
