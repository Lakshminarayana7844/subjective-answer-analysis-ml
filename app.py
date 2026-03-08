# app.py
# Flask backend for "Subjective Answer Analysis using Machine Learning"

from flask import Flask, render_template, request, redirect, url_for, session
from model import AnswerEvaluator

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret_key"

evaluator = AnswerEvaluator()

SAMPLE_QUESTIONS = [
    {
        "id": 1,
        "question": "What is Machine Learning?",
        "model_answer": (
            "Machine learning is a field of artificial intelligence that "
            "uses statistical techniques to give computer systems the ability "
            "to learn patterns from data and make predictions or decisions "
            "without being explicitly programmed for each task."
        )
    },
    {
        "id": 2,
        "question": "Explain the concept of overfitting in machine learning.",
        "model_answer": (
            "Overfitting occurs when a machine learning model learns not only "
            "the underlying patterns in the training data but also the noise. "
            "As a result, the model performs very well on the training set "
            "but poorly on new, unseen data because it fails to generalize."
        )
    },
    {
        "id": 3,
        "question": "What is the role of TF-IDF in text processing?",
        "model_answer": (
            "TF-IDF stands for Term Frequency–Inverse Document Frequency. "
            "It is a numerical statistic used to reflect how important a word "
            "is to a document in a collection. TF-IDF increases with the number "
            "of times a word appears in a document but is offset by how "
            "frequently the word appears in the entire corpus."
        )
    }
]

def get_question_by_id(qid: int):
    for q in SAMPLE_QUESTIONS:
        if q["id"] == qid:
            return q
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/exam", methods=["GET", "POST"])
def exam():
    if request.method == "GET":
        question = SAMPLE_QUESTIONS[0]
        return render_template("exam.html", question=question)

    if request.method == "POST":
        question_id = int(request.form.get("question_id"))
        student_answer = request.form.get("student_answer", "")

        question = get_question_by_id(question_id)
        if question is None:
            return redirect(url_for("home"))

        model_answer = question["model_answer"]
        result = evaluator.evaluate(model_answer=model_answer, student_answer=student_answer)

        session["question_text"] = question["question"]
        session["student_answer"] = student_answer
        session["model_answer"] = model_answer
        session["similarity"] = result["similarity"]
        session["marks"] = result["marks"]
        session["feedback"] = result["feedback"]

        return redirect(url_for("result"))

@app.route("/result")
def result():
    question_text = session.get("question_text")
    student_answer = session.get("student_answer")
    model_answer = session.get("model_answer")
    similarity = session.get("similarity")
    marks = session.get("marks")
    feedback = session.get("feedback")

    if question_text is None:
        return redirect(url_for("home"))

    return render_template(
        "result.html",
        question_text=question_text,
        student_answer=student_answer,
        model_answer=model_answer,
        similarity=similarity,
        marks=marks,
        feedback=feedback,
    )

if __name__ == "__main__":
    app.run(debug=True)
