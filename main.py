from preprocessing_data import preprocess_data
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LSTM-attack')

    parser.add_argument('-data', '--data_name', default='yelp', type=str,
                        help='Name of the data')

    args, unknown = parser.parse_known_args()

    # dataset_folders
    raw_data_path_train = "data/"+args.data_name+"/train.csv"
    raw_data_path_test = "data/"+args.data_name+"/test.csv"

    # preprocessing_data
    df_train, df_valid, df_test = preprocess_data(raw_data_path_train, raw_data_path_test, args.data_name).preprocessing()
    print(df_test[0:5])


