from flask import Flask, render_template, request

from sentimentAnalysis import tfidf_predict_rating_logistic, count_predict_rating_svm

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/start')
def start():
    return render_template('start.html')


@app.route('/startLogistic', methods=['POST'])
def start_logistic_post():
    mail = request.form['mail']
    unprocessed_review = [mail]
    predicted_rating = "Based on your review, your rating for this merchant or service should be around a " + str(int(tfidf_predict_rating_logistic(unprocessed_review)[0])) + " star! :)"
    return render_template('rating.html', review=mail, rating=predicted_rating)

@app.route('/startSVM', methods=['POST'])
def start_SVM_post():
    mail = request.form['mail']
    unprocessed_review = [mail]
    predicted_rating = "Based on your review, your rating for this merchant or service should be around a " + str(int(count_predict_rating_svm(unprocessed_review)[0])) + " star! :)"
    return render_template('rating.html', review=mail, rating=predicted_rating)

