from flask import Flask, render_template, request, send_file

from sentimentAnalysis import tfidf_predict_rating_logistic, count_predict_rating_svm, tfidf_predict_rating_svm
from statistics import mean

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/start')
def start():
    return render_template('start.html')


@app.route('/whitepaper')
def whitepaper():
    return render_template('whitepaper.html')


@app.route('/return-files/')
def return_files():
    try:
        return send_file('Sentiment_Analysis_Yelp_Reviews_Paper.pdf',
                         attachment_filename='whitepaper.pdf')
    except Exception as e:
        return str(e)


@app.route('/startLogisticTFIDF', methods=['POST'])
def start_logistic_post():
    mail = request.form['mail']
    unprocessed_review = [mail]
    predicted_rating = "Based on your review, your rating for this merchant or service should be around a " + str(
        int(tfidf_predict_rating_logistic(unprocessed_review)[0])) + " star! :)"
    return render_template('rating.html', review=mail, rating=predicted_rating)


@app.route('/startSVMCount', methods=['POST'])
def start_SVM_post():
    mail = request.form['mail']
    unprocessed_review = [mail]
    predicted_rating = "Based on your review, your rating for this merchant or service should be around a " + str(
        int(count_predict_rating_svm(unprocessed_review)[0])) + " star! :)"
    return render_template('rating.html', review=mail, rating=predicted_rating)


# @app.route('/startRFTFIDF', methods=['POST'])
# def start_RF_TFIDF_post():
#     mail = request.form['mail']
#     unprocessed_review = [mail]
#     predicted_rating = "Based on your review, your rating for this merchant or service should be around a " + str(int(tfidf_predict_rating_random_forest(unprocessed_review)[0])) + " star! :)"
#     return render_template('rating.html', review=mail, rating=predicted_rating)

@app.route('/startSVMTFIDF', methods=['POST'])
def start_SVM_TFIDF_post():
    mail = request.form['mail']
    unprocessed_review = [mail]
    predicted_rating = "Based on your review, your rating for this merchant or service should be around a " + str(
        int(tfidf_predict_rating_svm(unprocessed_review)[0])) + " star! :)"
    return render_template('rating.html', review=mail, rating=predicted_rating)


@app.route('/startAverage', methods=['POST'])
def start_avg_post():
    mail = request.form['mail']
    unprocessed_review = [mail]
    r = mean([tfidf_predict_rating_svm(unprocessed_review)[0], tfidf_predict_rating_logistic(unprocessed_review)[0],
              count_predict_rating_svm(unprocessed_review)[0]])
    predicted_rating = "Based on your review, your rating for this merchant or service should be around a " + str(
        round(r, 2)) + " star! :)"
    return render_template('rating.html', review=mail, rating=predicted_rating)
