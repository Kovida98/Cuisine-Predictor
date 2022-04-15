import pytest
from project2 import cuisine
input=['paprika', 'banana','rice krispies','plain flour', 'ground pepper']
N=3
id, cuisine_types, ingredients =cuisine.readdata('yummly.json')
list = cuisine.processingingredients(ingredients)
input_list = cuisine.processinginput(input, list)
vector=cuisine.countvectorizer(input_list)
X_train,X_test,y_train,y_test,ingredientsvector_data,inputvector_data=cuisine.traindata(vector,cuisine_types)
test,score= cuisine.knnmodel(X_train,X_test,y_train,y_test,ingredientsvector_data,inputvector_data)
result = cuisine.topNcuisines(inputvector_data,ingredientsvector_data,id,N)
result_json=cuisine.output_json(result,test,score)
def test_readdata():
    x,y,z=cuisine.readdata('yummly.json')
    assert x is not None
    assert y is not None
    assert z is not None
def test_processingingredients():
    # id,cuisine,ingredients=project2.readdata()
    # list=project2.processingingredients(ingredients)
    assert list is not None
def test_processinginput():
    # id, cuisine, ingredients = project2.readdata()
    # list = project2.processingingredients(ingredients)
    # input_list=project2.processinginput(input,list)
    assert input_list is not None
def test_countvectorizer():
    # id, cuisine, ingredients = project2.readdata()
    # list = project2.processingingredients(ingredients)
    # input_list = project2.processinginput(input, list)
    # vector=project2.countvectorizer(input_list)
    assert vector is not None
def test_traindata():
    assert X_train is not None

def test_knnmodel():
    assert score < 1

def test_topNcuisine():
    assert result.shape[0] == N

def test_output_json():
    dict={}
    assert result_json is not None