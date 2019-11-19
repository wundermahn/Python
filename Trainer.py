import pandas as pd, numpy as np, re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords

stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 
'again', 'there', 'about', 'once', 'during', 'out', 'very', 
'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 
'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 
'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 
'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 
'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 
'were', 'her', 'more', 'himself', 'this', 'down', 'should', 
'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 
'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 
'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 
'does', 'yourselves', 'then', 'that', 'because', 'what', 
'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 
'he', 'you', 'herself', 'has', 'just', 'where', 'too', 
'only', 'myself', 'which', 'those', 'i', 'after', 'few', 
'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 
'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

# This function removes numbers from an array
def remove_nums(arr): 
    # Declare a regular expression
    pattern = '[0-9]'  
    # Remove the pattern, which is a number
    arr = [re.sub(pattern, '', i) for i in arr]    
    # Return the array with numbers removed
    return arr

# This function cleans the passed in paragraph and parses it
def get_words(para):   
    # Split it into lower case    
    lower = para.lower().split()
    # Remove punctuation
    no_punctuation = (nopunc.translate(str.maketrans('', '', string.punctuation)) for nopunc in lower)
    # Remove integers
    no_integers = remove_nums(no_punctuation)
    # Remove stop words
    dirty_tokens = (data for data in no_integers if data not in stop_words)
    # Ensure it is not empty
    tokens = (data for data in dirty_tokens if data.strip())
    # Ensure there is more than 1 character to make up the word
    tokens = (data for data in tokens if len(data) > 1)
    
    # Return the tokens
    return tokens 

# Function to collect required F1, Precision, and Recall Metrics
def collect_metrics(actuals, preds):
    # Create a confusion matrix
    matr = confusion_matrix(actuals, preds, labels=[2, 4])
    # Retrieve TN, FP, FN, and TP from the matrix
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(actuals, preds).ravel()

    # Compute precision
    precision = true_positive / (true_positive + false_positive)
    # Compute recall
    recall = true_positive / (true_positive + false_negative)
    # Compute F1
    f1 = 2*((precision*recall)/(precision + recall))

    # Return results
    return precision, recall, f1

not_bot = pd.read_csv("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Project\Data\\Small\\NotBot.csv", skiprows=1)
bot = pd.read_csv("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Project\Data\\Small\\Bot.csv", skiprows=1)
csv_table = pd.DataFrame(np.vstack((not_bot.values, bot.values)))
csv_table.columns = ['username', 'tweet', 'following', 'followers', 'is_retweet', 'is_bot']

new_df = csv_table.groupby('username').head(20).reset_index(drop=True)

# Create the overall corpus
s = pd.Series(new_df['tweet'])
corpus = s.apply(lambda s: ' '.join(get_words(s)))

# Create a vectorizer
vectorizer = TfidfVectorizer()
# Compute tfidf values
# This also updates the vectorizer
test = vectorizer.fit_transform(corpus)

# Create a dataframe from the vectorization procedure
df = pd.DataFrame(data=test.todense(), columns=vectorizer.get_feature_names())

# Merge results into final dataframe
result = pd.concat([new_df, df], axis=1, sort=False)

labels = result['is_bot']

# https://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas
X_train, y_train, X_test, y_test = train_test_split(result[result.columns.difference(['is_bot', 'username', 'tweet'])], labels, test_size = 0.2)

knn = KNeighborsClassifier(n_neighbors = 7)
rf = RandomForestClassifier()
mlp = MLPClassifier()

knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
mlp.fit(X_train, y_train)

knn_preds = knn.predict(X_test)
rf_preds = rf.predict(X_test)
mlp_preds = mlp.predict(X_test)

knn_precision, knn_recall, knn_f1 = collect_metrics(knn_preds, y_test)
rf_precision, rf_recall, rf_f1 = collect_metrics(rf_preds, y_test)
mlp_precision, mlp_recall, mlp_f1 = collect_metrics(mlp_preds, y_test)

# Pretty print the results
print("KNN | Recall: {} | Precision: {} | F1: {}".format(knn_recall, knn_precision, knn_f1))
print("MLP     | Recall: {} | Precision: {} | F1: {}".format(mlp_recall, mlp_precision, mlp_f1))
print("RF      | Recall: {} | Precision: {} | F1: {}".format(rf_recall, rf_precision, rf_f1))