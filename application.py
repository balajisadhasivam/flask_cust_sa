from flask import Flask, request, render_template, jsonify
import pickle
from text_cleaning import text_data_cleaning
appliction = Flask(__name__)

with open('cust_review_senti.pkl', 'rb') as file:
    clf, loaded_text_data_cleaning = pickle.load(file)


@appliction.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@appliction.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
    #request_data = request.get_json()
        #review = request_data.get('review')
        review = request.form['review']
        cleaned_review = text_data_cleaning(review)
        sentiment = clf.predict([" ".join(cleaned_review)])[0]  # Predict sentiment for the cleaned review
        # Convert sentiment to a string before including it in the response
        sentiment_str = "Positive" if sentiment == 1 else "Negative"
        response = {
            "review": review,
            "sentiment": sentiment_str
        }

        return jsonify(response)

    except Exception as e:
        error_response = {"error": str(e)}
        return jsonify(error_response), 500

if __name__ == "__main__":
    appliction.run(debug=True)
