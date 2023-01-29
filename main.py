from flask import Flask, render_template, request, redirect
from transformers import pipeline
import sqlite3

app = Flask(__name__)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ['good poetry', 'bad poetry']

def get_db():
    conn = sqlite3.connect('submissions.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS submissions (submission_text TEXT, email TEXT, decision TEXT)''')
    return conn

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        submission_text = request.form['submission_text']
        email = request.form['email']
        if not submission_text or not email:
            error = 'Please fill out both the poem and email fields.'
            return render_template('submit.html', error=error)
        result = classifier(submission_text, candidate_labels)
        if result['labels'][0] == 'good poetry':
            decision = 'accepted'
        else:
            decision = 'denied'
        conn = get_db()
        conn.execute("INSERT INTO submissions (submission_text, email, decision) VALUES (?,?,?)",
                     (submission_text, email, decision))
        conn.commit()
        conn.close()
        return render_template('result.html', decision=decision)
    return render_template('submit.html')

if __name__ == '__main__':
    app.run()
