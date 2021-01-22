import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
import nltk

# For the first time
# nltk.download()

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

def remove_stopwords(tokenized_text):

    #Stop words
    stopwords = nltk.corpus.stopwords.words('english')
    #print(stopwords)
    text = "".join([word+" " for word in tokenized_text if word not in stopwords])
    return text

class preprocess_data:
     def __init__(self, train_data, test_data, data_name):
         self.train_data = train_data
         self.test_data = test_data
         self.data_name = data_name

     def preprocessing(self):

        if self.data_name == "yelp":

            # Read Raw data
            df_raw_train = pd.read_csv(self.train_data)
            df_raw_test = pd.read_csv(self.test_data)
            #print(df_raw_train.size) # 1119998
            #print(df_raw_test.size) #75998

            # add column headers to df
            df_raw_train.columns = ['label', 'review']
            df_raw_test.columns = ['label', 'review']

            # Remove punctuations
            df_raw_train["review"] = df_raw_train['review'].str.replace('[^\w\s]','')
            df_raw_test["review"] = df_raw_test['review'].str.replace('[^\w\s]','')
            #print(df_raw_train[0:5])
            #print(df_raw_train.size) #1119998
            #print(df_raw_test.size) #75998


            # Remoce stopwords
            df_raw_train['review'] = df_raw_train['review'].apply(lambda x: tokenize(x.lower())).apply(lambda x: remove_stopwords(x))
            df_raw_test['review'] = df_raw_test['review'].apply(lambda x: tokenize(x.lower())).apply(lambda x: remove_stopwords(x))
            #df_raw_train['review'] = df_raw_train['review'].apply(lambda x: remove_stopwords(x))
            #print(df_raw_train[0:5])
            #print(df_raw_train.size) #1119998
            #print(df_raw_test.size) #75998
            # Drop rows with empty text
            df_raw_train.drop( df_raw_train[df_raw_train.review.str.len() < 5].index, inplace=True)
            df_raw_test.drop( df_raw_test[df_raw_test.review.str.len() < 5].index, inplace=True)
            #print(df_raw_train[0:5])
            #print(df_raw_train[0:5])

            # Train_valid split
            df_train, df_valid = train_test_split(df_raw_train, train_size = 0.93, random_state = 1)
            df_test = shuffle(df_raw_test, random_state=1)
            print("Size of the training data =", len(df_train)) # 1041386
            print("Size of the validation data =",len(df_valid)) # 78384
            print("Size of the test data =",len(df_test)) # 75992
            print("preprocessed_data_without_punctuations_stopwords_emptyrows =\n", df_train[0:5])

            # save to CSV files in preprocessed folder
            df_train.to_csv('data/' + self.data_name + '/preprocessed_data/train.csv', index=False)
            df_valid.to_csv('data/' + self.data_name + '/preprocessed_data/valid.csv', index=False)
            df_test.to_csv('data/' + self.data_name + '/preprocessed_data/test.csv', index=False)

        return df_train, df_valid, df_test

