import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
import numpy as np
import pickle

nltk.download('stopwords')
nltk.download('punkt')
tfidf_vectorizer = pickle.load(open('./tfidf_vectorizer_Py2.sav', 'rb'))
clf400 = pickle.load(open('./trained_mult_logistic_reg_model_tfidf_Py2.sav', 'rb'))

count_vectorizer = pickle.load(open('./count_vectorizer_Py2.sav', 'rb'))
count_svm = pickle.load(open('./trained_svm_model_count_Py2.sav', 'rb'))

# tfidf_rf = pickle.load(open('./trained_random_forest_model_tfidf_Py2.sav', 'rb'))
tfidf_svm =  pickle.load(open('./trained_svm_model_tfidf_Py2.sav', 'rb'))


def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    """
    Input: str
    Output: str
    Removes stop words like I, me, the, etc. For preprocessing the data
    """
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_punctuation(data):
    """
    Input: str
    Output: str
    Further preprocessing
    """
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    """
    Input: str
    Output: str
    Further preprocessing
    """
    return np.char.replace(data, "'", "")


def stemming(data):
    """
    Input: str
    Output: str
    Converts words to their stem. Ex: worked -> work. Removes suffix and affix. No need for lemmatization for TFIDF
    """
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def preprocess(x):
    """
    Input: x, raw data in the form of an iterable of strings
    Output: Preprocessed X in the form an iterable of strings
    """

    x = map(lambda x: convert_lower_case(x), x)  # Convert each review to lowercase
    x = map(lambda x: remove_punctuation(x), x)  # Remove punctuation from each review
    x = map(lambda x: remove_apostrophe(x), x)  # Remove apostrophes from each review
    x = map(lambda x: remove_stop_words(x), x)  # Remove stop words from each review
    x = map(lambda x: convert_numbers(x), x)  # Convert numerics to string equivalents
    x = map(lambda x: stemming(x), x)  # Stem all the words from each review
    x = map(lambda x: remove_punctuation(x), x)  # Repeated just in case punctuation was reintroduced
    x = map(lambda x: convert_numbers(x), x)  # Just in case more numbers were reintroduced
    x = map(lambda x: stemming(x), x)  # Just in case numbers needed to be stemmed again
    x = map(lambda x: remove_punctuation(x), x)  # Repeated because num2words does give some hyphens and commas
    x = map(lambda x: remove_stop_words(x), x)  # Repeated because num2words does give stop words
    return list(x)


def tfidf_predict_rating_logistic(unpreprocessed_data):
    """
    trained_skmodel: the logistic regression trained model returned by sklearn (after fit)
    tfidf_vectorizer: the vectorizer used (after applying fit_transform). Used to convert unseen raw data to tfidf);
    unpreprocessed_data: an array or dataframe column of strings corresponding to unseen reviews
    """

    preprocessed_sample = preprocess(unpreprocessed_data)
    sample_tfidf = tfidf_vectorizer.transform(preprocessed_sample)
    return clf400.predict(sample_tfidf)

def count_predict_rating_svm(unpreprocessed_data):
    """
    trained_skmodel: the logistic regression trained model returned by sklearn (after fit)
    count_vectorizer: the vectorizer used (after applying fit_transform). Used to convert unseen raw data to count vector);
    unpreprocessed_data: an array or dataframe column of strings corresponding to unseen reviews
    """
    preprocessed_sample = preprocess(unpreprocessed_data)
    sample_count = count_vectorizer.transform(preprocessed_sample)
    return count_svm.predict(sample_count)

# def tfidf_predict_rating_random_forest(unpreprocessed_data):
#     """
#     trained_skmodel: the random forest (50 trees) trained model returned by sklearn (after fit)
#     count_vectorizer: the vectorizer used (after applying fit_transform). Used to convert unseen raw data to tfidf vector);
#     unpreprocessed_data: an array or dataframe column of strings corresponding to unseen reviews
#     """
#     preprocessed_sample = preprocess(unpreprocessed_data)
#     sample_count = tfidf_vectorizer.transform(preprocessed_sample)
#     return tfidf_rf.predict(sample_count)

def tfidf_predict_rating_svm(unpreprocessed_data):
    """
    trained_skmodel: the svm trained model returned by sklearn (after fit)
    count_vectorizer: the vectorizer used (after applying fit_transform). Used to convert unseen raw data to tfidf vector);
    unpreprocessed_data: an array or dataframe column of strings corresponding to unseen reviews
    """
    preprocessed_sample = preprocess(unpreprocessed_data)
    sample_count = tfidf_vectorizer.transform(preprocessed_sample)
    return tfidf_svm.predict(sample_count)


# if __name__ == '__main__':
#     print count_predict_rating_svm(
#                                     ['This food was amazing. I love it!', 'EWWWWW....Never coming back to this place!',
#                                    'The food was Okay',
#                                    'This place was wonderful! Located in downtown Houston off Dallas Street, this is a new restaurant that I think will be a success in our area. The service was great. Our waiter, Brandon, was awesome. He gave us some great suggestions for dinner and drinks. For appetizers, I ordered the tuna, oysters and cheese board and all of them were delicious!! The tuna sashimi was perfectly reddish pink and thinly sliced, it almost melted im my mouth! I wanted to lick the plate after the ponzo and sel gris, that so tasty. Next i had, oysters which were so fresh and the right size. I didn\'t get a chance to ask where the originated from but that didn\'t matter because it was a great dozen. Lastly, the charcuterie selection gave you an offering of cow, sheep and goat milk\'s cheese and a scoop or pear jelly, were the notable pieces. Now to the main dish, I ordered the Roasted Texas Redfish presented on the half shell with a lump of crab and gulf shrimp. My sides were crispy brussels and black truffle mac & cheese. Fun fact, the sides are served family style, gives you a chnace to share with your friends and try their food! My redfish was absolutely delicious and they surprisingly topped it off with some popcorn and it gave it some added texture and flavor. The dish had a  sauce pontchartrain that tied together the shrimp, crab and redfish without having an overbearing fishy taste to it. The shrimp were huge and I had enough to take home for lunch the next day!! I totally recommend Guard & Grace to anyone looking for that new spot for happy hour or   your wanting to try some delicious food. I can see myself coming here again for date night!!! You have to give this place a try because it was awesome! It\'s Houston newest spot!! Bianca',
#                                    'ANYONE THAT TRIES TO SELL FAJITAS IN SOUTH TEXAS FOR $400.00 A PERSON IS EITHER AN IDIOT OR A POMPAUS JERK.  I AM sure this overpriced arrogant "restaurant" will meet the social needs of some Houstonians, but I am also just as sure for the vast eating out public in Houston,  this overpriced place to eat will be avoided like the plague!!'])
#     print tfidf_predict_rating_logistic(['This food was amazing. I love it!', 'EWWWWW....Never coming back to this place!',
#                                    'The food was Okay',
#                                    'This place was wonderful! Located in downtown Houston off Dallas Street, this is a new restaurant that I think will be a success in our area. The service was great. Our waiter, Brandon, was awesome. He gave us some great suggestions for dinner and drinks. For appetizers, I ordered the tuna, oysters and cheese board and all of them were delicious!! The tuna sashimi was perfectly reddish pink and thinly sliced, it almost melted im my mouth! I wanted to lick the plate after the ponzo and sel gris, that so tasty. Next i had, oysters which were so fresh and the right size. I didn\'t get a chance to ask where the originated from but that didn\'t matter because it was a great dozen. Lastly, the charcuterie selection gave you an offering of cow, sheep and goat milk\'s cheese and a scoop or pear jelly, were the notable pieces. Now to the main dish, I ordered the Roasted Texas Redfish presented on the half shell with a lump of crab and gulf shrimp. My sides were crispy brussels and black truffle mac & cheese. Fun fact, the sides are served family style, gives you a chnace to share with your friends and try their food! My redfish was absolutely delicious and they surprisingly topped it off with some popcorn and it gave it some added texture and flavor. The dish had a  sauce pontchartrain that tied together the shrimp, crab and redfish without having an overbearing fishy taste to it. The shrimp were huge and I had enough to take home for lunch the next day!! I totally recommend Guard & Grace to anyone looking for that new spot for happy hour or   your wanting to try some delicious food. I can see myself coming here again for date night!!! You have to give this place a try because it was awesome! It\'s Houston newest spot!! Bianca',
#                                    'ANYONE THAT TRIES TO SELL FAJITAS IN SOUTH TEXAS FOR $400.00 A PERSON IS EITHER AN IDIOT OR A POMPAUS JERK.  I AM sure this overpriced arrogant "restaurant" will meet the social needs of some Houstonians, but I am also just as sure for the vast eating out public in Houston,  this overpriced place to eat will be avoided like the plague!!'])