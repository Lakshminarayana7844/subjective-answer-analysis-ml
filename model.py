# model.py
# Machine learning logic for subjective answer evaluation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AnswerEvaluator:
    """Evaluator that compares student answers with model answers using TF-IDF and cosine similarity."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

    def evaluate(self, model_answer: str, student_answer: str) -> dict:
        """Compute similarity and marks for student answer."""
        
        if not student_answer or student_answer.strip() == "":
            return {
                "similarity": 0.0,
                "marks": 0.0,
                "feedback": "You did not provide any answer. Please try to write something."
            }

        corpus = [model_answer, student_answer]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_score = float(sim_matrix[0][0])
        raw_marks = similarity_score * 10.0
        marks = max(0.0, min(10.0, raw_marks))

        if similarity_score >= 0.8:
            feedback = "Excellent answer! You covered most of the key points."
        elif similarity_score >= 0.6:
            feedback = "Good attempt. You captured many important ideas, but there is room for improvement."
        elif similarity_score >= 0.4:
            feedback = "Fair answer. You mentioned some relevant points; try to be more complete and precise."
        elif similarity_score > 0:
            feedback = "Your answer is only slightly related to the expected one. Please review the topic and try again."
        else:
            feedback = "Your answer does not match the expected answer. Please study the material and attempt again."

        return {
            "similarity": similarity_score,
            "marks": round(marks, 2),
            "feedback": feedback
        }
