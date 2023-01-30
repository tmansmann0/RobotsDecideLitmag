from flask import Flask, render_template, request, redirect
from transformers import pipeline
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask_basicauth import BasicAuth

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'username'
app.config['BASIC_AUTH_PASSWORD'] = 'password'
basic_auth = BasicAuth(app)

#define classifier for determining what is accepted
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ['good poetry', 'bad poetry']

# Create an SQLAlchemy base
Base = declarative_base()

# Define the Submission table
class Submission(Base):
    __tablename__ = 'submissions'
    id = Column(Integer, primary_key=True)
    submission_text = Column(String)
    email = Column(String)
    decision = Column(String)

def get_db():
    # Create a SQLite database
    engine = create_engine('sqlite:///submissions.db', echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result')
def result():
    decision = request.args.get('decision')
    return render_template('result.html', decision=decision)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        submission_text = request.form['submission_text']
        email = request.form['email']
        return render_template('loading.html')
        result = classifier(submission_text, candidate_labels)
        if result['labels'][0] == 'good poetry':
            decision = 'accepted'
        else:
            decision = 'denied'
        session = get_db()
        # Create a new Submission object and add it to the session
        submission = Submission(submission_text=submission_text, email=email, decision=decision)
        session.add(submission)
        # Commit the transaction
        session.commit()
        session.close()
        return redirect('/result')
    return render_template('submit.html')

@app.route('/submissions')
@basic_auth.required
def submissions():
    session = get_db()
    # Fetch all the submissions from the table
    submissions = session.query(Submission).all()
    session.close()

    # Replace newline characters with <br> tags
    for submission in submissions:
        submission.submission_text = submission.submission_text.replace('\n', '<br>')

    return render_template('submissions.html', submissions=submissions)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run()
