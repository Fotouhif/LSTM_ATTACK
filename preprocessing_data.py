import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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

            # add column headers to df
            df_raw_train.columns = ['label', 'review']
            df_raw_test.columns = ['label', 'review']

            # Remove punctuations
            df_raw_train["review"] = df_raw_train['review'].str.replace('[^\w\s]','')
            df_raw_test["review"] = df_raw_test['review'].str.replace('[^\w\s]','')


            # Drop rows with empty text
            df_raw_train.drop( df_raw_train[df_raw_train.review.str.len() < 5].index, inplace=True)
            df_raw_test.drop( df_raw_test[df_raw_test.review.str.len() < 5].index, inplace=True)
            #print(df_raw_train[0:5])


            # Train_valid split
            df_train, df_valid = train_test_split(df_raw_train, train_size = 0.8, random_state = 1)
            df_test = shuffle(df_raw_test, random_state=1)
            #print(df_train[0:5])

            df_train.to_csv('data/' + self.data_name + '/preprocessed_data/train.csv', index=False)
            df_valid.to_csv('data/' + self.data_name + '/preprocessed_data/valid.csv', index=False)
            df_test.to_csv('data/' + self.data_name + '/preprocessed_data/test.csv', index=False)

        return df_train, df_valid, df_test

