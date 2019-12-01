from flask import Flask, render_template, request

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
    return render_template('rating.html', rating = mail)

