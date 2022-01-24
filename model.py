import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommendationSystem:

    def __init__(self):
        # Load recommendation file
        self.recommendation_df = pd.read_pickle("./models/item_recommendation.pkl")

        # Reading Preprocessed data file
        self.ratings = pd.read_csv('./data/sample30_clean.csv' , encoding='latin-1', index_col=0)

        # load the vectorizer file
        self.word_vectorizer = pickle.load(open('./models/word_vectorizer.pkl', 'rb'))

        # load sentiment model file
        self.loaded_model = pickle.load(open("./models/model_sentiment.pkl", "rb"))

    def recommend_products(self, username):
        top_product = pd.DataFrame()
        error = None

        ## check the username is present in the given dataset or not
        if username.lower() not in self.ratings['reviews_username'].str.lower().values:
            return None

        try:
            product = self.recommendation_df.loc[username].sort_values(ascending=False)[0:20]
            recommendated_product = pd.DataFrame(product)
            
             #predict sentiment function is used to predict the sentiment for the given text
            def predict_sentiment(reviews_text):
                test_data = TfidfVectorizer(vocabulary=self.word_vectorizer.get_feature_names_out())
                X_test_value = test_data.fit_transform(reviews_text)
                sentiment_val= ''
                sentiment_val = pd.DataFrame(self.loaded_model.predict(X_test_value), columns=['sentiment'])
                reviews_text.reset_index(drop=True, inplace=True)
                sentiment_val.reset_index(drop=True, inplace=True) 
                product_sentiment = pd.concat([reviews_text, sentiment_val], axis=1)

                return product_sentiment

            
            def recommendate_product():
                selected_reviews_list = pd.DataFrame()
                for index, row in recommendated_product.iterrows():
                    product_filter = self.ratings["name"].isin([row.name])
                    selected_items = self.ratings[product_filter]
                    selected_reviews = predict_sentiment(selected_items.reviews_text)
                    selected_reviews["name"] = row.name
                    selected_reviews_list = selected_reviews_list.append(selected_reviews)
       
                return selected_reviews_list
            
            recommendated_product_review = recommendate_product()

            def filter_top5_product():
                sentiment_count = recommendated_product_review.groupby(['name','sentiment']).size().reset_index(name='counts')
                total_product = recommendated_product_review.groupby(['name']).size().reset_index(name='total_counts')
                product_sentiment = pd.merge(sentiment_count,total_product[['name','total_counts']],on='name', how='right')
                product_sentiment['percentage'] = (product_sentiment['counts']/product_sentiment['total_counts'])*100
                positive_product_sentiment = product_sentiment[(product_sentiment['sentiment'] == 1) & (product_sentiment['percentage'] >=30) ]
                top5_positive_product = positive_product_sentiment.sort_values('percentage',ascending=False)[0:5]

                return top5_positive_product


            top5_product  = filter_top5_product()
            top5_product.reset_index(inplace = True)
            top_product = pd.DataFrame(top5_product['name'])
            print(top_product)

        except KeyError:
            print("Inside Error")
            error = 'The Given username is not present in the dataset. Request you to provide the available username'
            return error

        
        #send to calling program
        return top_product



