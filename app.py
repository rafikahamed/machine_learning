from flask import Flask, render_template, request
from model import RecommendationSystem
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)  # intitialize the flaks app  # common

#Instance of the recommendation object
recommendation_system = RecommendationSystem()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/Submit",  methods=['POST'])
def Submit():
    error = None
    if request.method == 'POST':

        if request.form['username']:
            print(request.form['username'])
            show_flag =False
            error = None
            
            ## Calling the Model File
            top_product = recommendation_system.recommend_products( request.form['username'])

            if top_product is None:
                error = 'The Given username is not present in the dataset. Request you to provide the available username'
                return render_template("index.html", column_names=None, 
                        row_data=None, zip=zip, show_content = show_flag, error=error)
            else:
                show_flag = True
                error = None
                return render_template("index.html", column_names=top_product.columns.values, 
                row_data=list(top_product.values.tolist()), zip=zip, show_content = show_flag, error=error)


if __name__ == '__main__':
    app.run()
