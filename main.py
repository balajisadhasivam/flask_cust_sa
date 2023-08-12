from flask import Flask, request, render_template, jsonify
import pickle
from text_cleaning import text_data_cleaning
app = Flask(__name__)

with open('cust_review_senti.pkl', 'rb') as file:
    clf, loaded_text_data_cleaning = pickle.load(file)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        review = request.form['review']  # Extract review from the form data
        cleaned_review = text_data_cleaning(review)
        sentiment = clf.predict([" ".join(cleaned_review)])[0]  # Predict sentiment for the cleaned review

        # Convert sentiment to a string before including it in the response
        sentiment_str = "Positive" if sentiment == 1 else "Negative"

        response = {
            "review": review,
            "sentiment": sentiment_str
        }

        # Create a JSON response using the jsonify function
        return jsonify(response)

    except Exception as e:
        # Handle errors and return an error response
        error_response = {"error": str(e)}
        return jsonify(error_response), 500


if __name__ == "__main__":
    app.run(debug=True)
