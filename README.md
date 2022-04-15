## CUISINE PREDICTOR

The goal of this project is to get understanding of
the large menu set better by providing a cuisine predictor.
we are given the master list of all possible dishes, their ingredients,
an identifier, and the cuisine for thousands of different dishes. 
we have to realize that if you cluster foods by their ingredients you can help 
the restaurant change foods but keep the ingredients constant. 
we present a display of clustered ingredients and train a classifier
to predict the cuisine type of a new food.
The data sets for this project is provided by Yummly.com .

In this project,following steps are followed:
1. Train the food data with the provided dataset.
2. By taking the input of all ingredients given by user through command
   line argument.
3. By using model,i predicted the cuisine type.
4. Returning the top N closest foods for which N is defined in command line argument.

## AUTHOR:

Kovida Mothukuri

Author Email: kovida.mothukuri-1@ou.edu


## TREE STRUCTURE:

## PACKAGES USED:
1. json
2. os
3. numpy
4. pandas
5. nltk
6. sklearn

## FUNCTIONS:
I made a folder called project2 with two files titled `project2.py` and
`cuisine.py` in it.
I've written code in the `cuisine.py` file. I called
all these functions which wrote in `cuisine.py` in `project2.py` file.

### cuisine.py:
1. **readdata():**
   This function takes the `yummly.json` as input and returns 3 lists of
   id, cuisine type,ingredients. Here i am reading the yummly.json file 
   and appending id's ,cuisine type, ingredients in to separate lists named
   `id`,`cuisine type`,`ingredients_list`.
2. **processingingredients():**
   This function takes `ingredients_list` as input and returns a list named
   `list`.Here, i used porter stemmer method to stemmetize the ingredients
   list and i removed commas in the ingredients list and presented ingredinets
   as a string , simply words separated by spaces.
3. **processinginput():**
   This function takes the input of ingredients which are given from command
   line and in this function i am preprocessing data and stemmitizing 
   in the same way as ingredients list and appending that input string
   in to the processed ingredients list named `list` at the end and returns the `list`.
4. **countvectorizer():**
   This function takes the input as `list`. Here i used count vectorizer 
   to convert the words present in `list` to features.Each unique word
   in the document is turned into a feature, with a "1" assigned to 
   occurrences that include that word or feature and a "0" allocated
   to others. All words receive the same amount of weight and returns 
   a list of 1's and 0,s in list named `vector`.
5. **traindata():**
   This function takes the input as `vector` and list `cuisine_type`.
   Here, i am splitting the data `vector` in to two parts. I am taking the
   last element in the data `vector` and named it as `inputvector_data`
   which is the vectorized input given by the user from command line.
   The rest of the elements in data `vector` is named as `ingredientsvector_data`
   which is the vectorized ingredients list taken from yummly.json file.
   In this function, i am training the data `ingredientsvector_data` and `cuisine_type`
   using train_test_split method and this function returns `X_train,X_test,y_train,y_test,ingredientsvector_data,inputvector_data`.
6. **knnmodel():**
   This function takes `X_train,X_test,y_train,y_test,ingredientsvector_data,inputvector_data`
   as input.In this function, i used K-Nearest neighbours algorithm to predict
   the cuisine type using the trained data. Here i am checking the score
   and returning the predicted cuisine and score.
7. **topNcuisines():**
   This function takes `inputvector_data,ingredientsvector_data,id,N` as input
   where N is the number which is taken from command line. N represents 
   the Top N closest foods.In this function i am using cosine similarity
   between `inputvector_data,ingredientsvector_data` to get the similarity
   scores which is named as `sim`.I made data frame using `sim` and `id`
   values and takes the top N rows by sorting. This function returns the 
   dataframe which has top N closest scores named as `result`.
8. **output_json():**
   This function takes `result,test,score`.In this function, i changed the 
   data frame `result` in to json format. By doing some changes i am printing the
   predicted cuisine,score and top N cuisines in json format.

### project2.py:
In project2.py file, i have added parser arguments for Top N and ingredients flags.
Here in this function, i am calling each and every function which is written 
in `cuisine.py` file.I created a function names main. In that function, i am 
calling six functions which are written in cuisine.py file. This function returns
`inputvector_data,ingredientsvector_data,id,predicted_cuisine,score`.
By using this returned elements, i am calling cuisine.topNcuisines() method
and cuisine.output_json() methods to print the desired output.

## TEST CASES:
For test cases, i created a folder named `tests` and in that created a file
named `test_all.py`. I kept file called `yummly.json` in this folder.
In test_all.py file, i called each function written in cuisine.py file globally.
So that there will be no calling of those functions so many times.
1. **test_readdata():**
   This function tests the `readdata()` function which returns the id, cuisine_type,
   ingredients_list. I wrote an assert statement to check whether the id,cuisine_type,
   ingredients_list is not none.
2. **test_processingingredients():**
   This function tests the `processingingredients()` function which returns
   processed list named `list`.I wrote an assert statement to check whether the
   `list` is not none.
3. **test_processinginput():**
   This function tests the `processinginput()` function which returns
   processed list named `input_list`.I wrote an assert statement to check whether the
   `input_list` is not none.
4. **test_countvectorizer():**
   This function tests the `countvectorizer()` function which returns
   vectorized data named `vector`.I wrote an assert statement to check whether the
   `vector` is not none.
5. **test_traindata():**
   This function tests the `traindata()` function which returns
   splitted data `X_train,X_test,y_train,y_test,ingredientsvector_data,inputvector_data`.
   I wrote an assert statement to check whether the `X_train` is not none.
6. **test_knnmodel():**
   This function tests the `knnmodel()` function which returns
   predicted cuisine and score named `test,score`.I wrote an assert statement to check whether the
   `score` is less than one.
7. **test_topNcuisine():**
   This function tests the `topNcuisines()` function which returns
   data frame named `result`.I wrote an assert statement to check whether the
   number of rows in `result` are equal to N.
8. **test_output_json():**
   This function tests the `output_json()` function which returns
   result_json .I wrote an assert statement to check whether the
   `result_json` is not none.

## COMMANDS TO RUN:
Here to run the file we have to use below command:

`pipenv run python project2.py 
--ingredient paprika --ingredient banana --ingredient "rice Krispies" 
--N 5`

To run test cases we can use any one of the following commands

`pipenv run python -m pytest`

**Expected Output:**

`{'cuisine': 'japanese', 'score': 0.5, 'Closest': '[{"id":26401,"score":0.3333333333},{"id":8882,"score":0.3333333333},{"id":28079,"score":0.2886751346},{"id":45820,"score":0.2886751346},{"id":30371,"score":0.2886751346}]'}`

## ASSUMPTIONS AND BUGS:
To run the code using command, You have to give ingredients flag first and
N flag next. My code will not run, if you give N flag first and ingredients flag next. 
While running the code, i am getting some warnings related to deprecated issues,
please ignore such warnings.
## DIRECTIONS TO INSTALL:

You can create folders and files using mkdir and touch commands.
Here in this project we will be using python 3.10.2 version. to install that use this command.

`pipenv install --python 3.10.2`

After downloading the project from github, go to that directory using cd.Install pipenv by using
command. `pip install pipenv`. After that by checking requirements.txt file, you have to install all
required packages.  you need to install pytest using this command `pipenv install pytest`.Once the installation of pytest is done, you will be able to
run the unittests using `pipenv run python -m pytest`. 

you can run the code using
`pipenv run python project2.py 
--ingredient paprika --ingredient banana --ingredient "rice Krispies" 
--N 5`.

## EXTERNAL LINKS USED:

[https://www.geeksforgeeks.org/read-json-file-using-python/](https://www.geeksforgeeks.org/read-json-file-using-python/)
[https://riptutorial.com/nltk/example/27393/porter-stemmer](https://riptutorial.com/nltk/example/27393/porter-stemmer)
[https://investigate.ai/text-analysis/counting-words-with-scikit-learns-countvectorizer/](https://investigate.ai/text-analysis/counting-words-with-scikit-learns-countvectorizer/)
[https://data-flair.training/blogs/machine-learning-algorithms-in-python/](https://data-flair.training/blogs/machine-learning-algorithms-in-python/)
[https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_knn_algorithm_finding_nearest_neighbors.htm](https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_knn_algorithm_finding_nearest_neighbors.htm)
[https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html)

