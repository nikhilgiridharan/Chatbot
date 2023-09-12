import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you"],
        "responses": ["Hi there", "Hello", "Hey", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot"],
        "responses": ["You are welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you"],
        "responses": ["I am a chatbot", "My purpose is to help you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I cannot provide weather information."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "How do I create a budget"],
        "responses": ["To make a budget, track income and expenses. Put your income towards essential expenses. Next, allocate some of your income towards savings and debt repayment. Finally, the remainder of your income goes toward discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["Credit score is a number that is based on your credit history and is used by lenders to determine whether to lend you money. The higher your score, the more likely you are to be approved for credit."]
    }
]



vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tag = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tag.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tag
clf.fit(x, y)

def chatbot(inputTxt):
    inputTxt = vectorizer.transform([inputTxt])
    tag = clf.predict(inputTxt)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response


counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Type a message and Enter to start a conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting.")
            st.stop()

if __name__ == '__main__':
    main()