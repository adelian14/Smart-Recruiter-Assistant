# run.py
from app.chatbot import answer_question

if __name__ == "__main__":
    mock_context = """
    - Ahmed Hassan graduated from Cairo University in 2020.
    - He worked on time series forecasting using LSTM models.
    - Skills: Python, TensorFlow, Time Series Analysis, SQL
    - Mona Ahmed graduated from Cairo University in 2020.
    - He worked on frontend project using React.
    - Skills: React, JavaScript, SQL
    """
    q = "Who has experience in time series?"
    print(answer_question(q, mock_context))
