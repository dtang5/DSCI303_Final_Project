from flask import Flask, render_template, request

from sentimentAnalysis import tfidf_predict_rating_logistic

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/start')
def start():
    return render_template('start.html')


@app.route('/start', methods=['POST'])
def start_post():
    mail = request.form['mail']
    unprocessed_review = [mail]
    predicted_rating = "Based on your review, your rating for this merchant or service should be around a " + str(int(tfidf_predict_rating_logistic(unprocessed_review)[0])) + " star! :)"
    return render_template('rating.html', review=mail, rating=predicted_rating)

