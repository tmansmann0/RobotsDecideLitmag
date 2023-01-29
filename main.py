from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        submission_text = request.form["submission_text"]
        email = request.form["email"]
        # Validation to ensure both fields are filled out
        if not submission_text or not email:
            error = "Please fill out both fields before submitting."
            return render_template("submit.html", error=error)
        else:
            # Classifier code goes here
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            candidate_labels = ['good poetry', 'bad poetry']
            result = classifier(submission_text, candidate_labels)
            if result['labels'][0] == 'good poetry':
                decision = 'accepted'
            else:
                decision = 'denied'
            # Save submission, email, and decision to a database
            # Return the decision to the user
            return render_template("result.html", decision=decision)
    else:
        return render_template("submit.html")

if __name__ == "__main__":
    app.run(debug=True)
