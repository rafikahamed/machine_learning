from flask import Flask, render_template
from flask import request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/Submit",  methods=['POST'])
def Submit():
    error = None
    if request.method == 'POST':
        if request.form['username']:
            usernmae = ''
            print(request.form['username'])
            usernmae = request.form['username']

            # Loadrecommendation file
            recommendation_df = pd.read_pickle("./models/item_recommendation.pkl")

            # Reading Actual Data file
            ratings = pd.read_csv('./data/sample30.csv' , encoding='latin-1')

            # load the vectorizer
            loaded_vectorizer = pickle.load(open('./models/vectorizer.pkl', 'rb'))

            # load model from file
            loaded_model = pickle.load(open("./models/sentiment_analysis.pkl", "rb"))

            product = recommendation_df.loc[usernmae].sort_values(ascending=False)[0:20]
            recommendated_product = pd.DataFrame(product)

            def predict_sentiment(reviews_text):
                test_data = CountVectorizer(vocabulary=loaded_vectorizer.get_feature_names_out())
                X_test_value = test_data.fit_transform(reviews_text)
                sentiment_val= ''
                sentiment_val = pd.DataFrame(loaded_model.predict(X_test_value), columns=['sentiment'])
                reviews_text.reset_index(drop=True, inplace=True)
                sentiment_val.reset_index(drop=True, inplace=True) 
                product_sentiment = pd.concat([reviews_text, sentiment_val], axis=1)
                return product_sentiment

            
            def value_cnt():
                selected_value_cnt = pd.DataFrame()
                for index, row in recommendated_product.iterrows():
                    product_filter = ratings["name"].isin([row.name])
                    selected_items = ratings[product_filter]
                    selected_reviews = predict_sentiment(selected_items.reviews_text)
                    df = selected_reviews['sentiment'].value_counts(normalize=True).rename_axis('sentiment_values').reset_index(name='counts')
                    df["product"] = row.name
                    selected_value_cnt = selected_value_cnt.append(df)
                
                return selected_value_cnt
            
            selected_value_cnt_review = value_cnt()

            def filter_top_product():
                filter_positive_records = selected_value_cnt_review[selected_value_cnt_review['sentiment_values'] == 1]
                top5_positive_product = filter_positive_records.sort_values('counts',ascending=False)[0:20]
                return top5_positive_product

            top_product_details = filter_top_product()
            top_product_details.reset_index(inplace = True)
            top_product = pd.DataFrame(top_product_details['product'])
            show_flag = True
            print(top_product)

    return render_template("index.html", column_names=top_product.columns.values, row_data=list(top_product.values.tolist()), zip=zip, show_content = show_flag)


if __name__ == '__main__':
    app.run()
