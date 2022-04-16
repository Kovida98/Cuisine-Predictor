import argparse
import cuisine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--N",type=str, required= False,help= 'used to get top N cuisines')
    parser.add_argument("--ingredient", type=str, required=True,action="append",
                            help="to get the input ingredients")


    args = parser.parse_args()
    if args.ingredient:
        id, cuisine_type, ingredients_list = cuisine.readdata('yummly.json')

        processed_list = cuisine.processingingredients(ingredients_list)

        stemmer_list = cuisine.processinginput(args.ingredient, processed_list)

        countvector = cuisine.countvectorizer(stemmer_list)

        X_train, X_test, y_train, y_test, ingredientsvector_data, inputvector_data = cuisine.traindata(countvector,
                                                                                                       cuisine_type)

        predicted_cuisine, score = cuisine.knnmodel(X_train, X_test, y_train, y_test, ingredientsvector_data,inputvector_data)
        #inputvector_data, ingredientsvector_data, id,predicted_cuisine,score = main(args.ingredient)
    if args.N:
        result = cuisine.topNcuisines(inputvector_data, ingredientsvector_data, id, args.N)
        output= cuisine.output_json(result,predicted_cuisine,score)