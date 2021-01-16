# LSTM_NLP

Training a bidirectional LSTM model for NLP datasets by using torchtext.

The sampled data in this repository is the smaller version of YELP polarity dataset which you can download the full version from here.

After running the main.py file, the processed data will be saved in the /data/"dataset_name"/preprocessed_data.
The trained model will be saved in the /model directory.


### How to train the model? Run "main.py" file.

* ```-data```: data in the /data directory which you would like to train
* ```-batch_size```: batch size for training
* ```-num_epochs```:  number of epochs
* ```-lr```: learning rate for training

### Train the model

```bash
  $ python main.py 
```

### Packages

argparse
os
torch
torchtext
pandas
sklearn





