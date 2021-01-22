from preprocessing_data import preprocess_data
from lstm_train import train_model
import argparse
import os
import torch

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("The code is running on", device)

    parser = argparse.ArgumentParser(description='LSTM-attack')

    parser.add_argument('-data', '--data_name', default='yelp', type=str,
                        help='Name of the data')
    parser.add_argument('-batch_size', '--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('-lr', '--lr', default=0.01, type=int,
                        help='learning rate')
    parser.add_argument('-num_epochs', '--num_epochs', default=10, type=int,
                        help='number of epochs')

    args, unknown = parser.parse_known_args()
    #print(args.num_epochs)

    #make preprocessed data for the first time
    if os.path.isdir('data/'+args.data_name+'/preprocessed_data') ==False:
        # dataset_folders
        raw_data_path_train = "data/"+args.data_name+"/train.csv"
        raw_data_path_test = "data/"+args.data_name+"/test.csv"

        os.mkdir(os.path.join('data/'+args.data_name+'/', 'preprocessed_data'))

        # preprocessing_data
        df_train, df_valid, df_test = preprocess_data(raw_data_path_train, raw_data_path_test, args.data_name).preprocessing()
        #print(df_test[0:5])

    source_folder = "data/"+args.data_name+"/preprocessed_data"
    train_model(source_folder, args.batch_size, device, args.num_epochs, args.lr).model_training()


### How to run

# python main.py --num_epochs 20



