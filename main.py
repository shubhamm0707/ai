import nltk
import numpy as np
import random
import string

# Define the corpus of training data
corpus = {
     'seriously' : ['yes'],
     'acha suno' : ['bolo', 'hn bolo na', 'yes mam boliye'],
     'tabyt theek hai tumhari' : ['hn', 'hn yr theek hai', 'yes bilkul', 'tumse bat ho gyi to achi ho gyi yr'],
     'or btao na' : ["missing you yr"],
    'kya kar rhe ho' : ['kuch nhi yr bs ese hi', 'bs ese hi vella panti', 'kaam kr rha hun yr'],
    'dinner ho gya' : ['hn ho gya, tumne kuch khaya', 'hn just abhi kiya', 'hn abhi khaya'],
    'ok' : ['hnm']
}


# Define a function to preprocess the training data
def preprocess(text):
    # Convert the text to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([c for c in text if c not in string.punctuation])
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    # tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    # Stem the words
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join the tokens back into a string
    return ' '.join(tokens)

# Preprocess the training data
preprocessed_corpus = {}
similarities = {}
for intent, sentences in corpus.items():
    preprocessed_corpus[intent] = [preprocess(sentence) for sentence in sentences]

def generate_response(text):
    # Preprocess the user's input
    text = preprocess(text)
    # Compute the similarity between the user's input and the training data
    similarities = {}
    for intent, sentences in preprocessed_corpus.items():
        similarity = 1 - nltk.jaccard_distance(set(intent.split()), set(text.split()))
        similarities[intent] = similarity
    # Return the most similar intent
    intent = max(similarities, key=similarities.get)
    # Return a random response from the corresponding intent
    return random.choice(corpus[intent])


# Define a function to generate a response to a user's input


# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = generate_response(user_input)
    print("Bot:", response)
