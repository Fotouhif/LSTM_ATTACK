import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


source_folder = "data/preprocessed_data"
model_folder = "model"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("The code is running on", device)

# Fields
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
#print(label_field)
text_field = Field(tokenize=None, lower=True, include_lengths=True, batch_first=True)
fields = [('label', label_field), ('review', text_field)]
#print(fields)

# TabularDataset # Defines a Dataset of columns stored in CSV, TSV, or JSON format.
train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv', test='test.csv',format='CSV', fields=fields, skip_header=True)
#print(train)
#print(valid)
#print(test)

print(train[3].label)
print(train[3].review[:10])

# Iterators
train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.review), device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.review), device=device, sort=True, sort_within_batch=True)
test_iter = BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.review), device=device, sort=True, sort_within_batch=True)

# Vocabulary# Torchtext handles mapping words to integers, but it has to be told the full range of words it should handle.
# In our case, we probably want to build the vocabulary on the training set only, so we run the following code:
text_field.build_vocab(train,vectors = 'glove.6B.200d') # Count the frequencies of tokens in all documents and build a vocab using the tokens frequencies
#print(text_field.vocab)
#print(len(text_field.vocab))

class LSTM(nn.Module):

    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300) #an Embedding module containing 676135 tensors of size 300
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.2)

        self.fc = nn.Linear(2*dimension, 2)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)
        #text_emb = self.drop(text)
        #print("text_embedding =", text_emb.size())
        #print("text_len =", text_len.size())
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        #print("packed_input =", packed_input)
        packed_output, (ht, ct) = self.lstm(packed_input)
        #print("packed_output =", packed_output)
        #print("ht =", ht[:,-1,:].size())
        #print("ct =", ct.size())
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        #print("output_size =", output.size())

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        out_reduced = self.drop(out_reduced)
        #print("out_reduced_size =",out_reduced.size())
        
        text_fea = self.fc(out_reduced)
        return text_fea


# Save and Load Functions

def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def train(model,
          optimizer,
          criterion = nn.CrossEntropyLoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 10,
          eval_every = len(train_iter) // 2,
          file_path = model_folder,
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        #print("Epoch ", epoch)
        #for label,_ in train_loader:
        for data_tensors,_ in train_loader:
          #print("step_number =", i)
          label = data_tensors[0]-1
          #print("label =", label)
          #print("label size=", label.size())
          review = data_tensors[1][0]
          review_len=data_tensors[1][1]
 
          label = label.to(device,dtype=torch.long)
          review = review.to(device)
          #print("namedesc_size =", namedesc.size())
          review_len = review_len.to(device)
          output = model(review, review_len)
          #print("output model size = ",output.size())

          loss = criterion(output, label)
          #print("loss =", loss)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # update running values
          running_loss += loss.item()
          global_step += 1

          # evaluation step
          if global_step % eval_every == 0:
              #print("Evaluation starts")
              model.eval()
              with torch.no_grad():                    
                # validation loop
                for data_tensors,_ in valid_loader:
                  label = data_tensors[0]-1
                  review = data_tensors[1][0]
                  review_len=data_tensors[1][1]

                  label = label.to(device,dtype=torch.long)
                  review = review.to(device)
                  #print("label_size_Eval =", label.size())
                  #print("namedesc_size_eval =", namedesc.size())
                  review_len = review_len.to(device)
                  #print("namedesc_len_size =",namedesc_len.size())
                  output = model(review, review_len)

                  loss = criterion(output, label)
                  valid_running_loss += loss.item()

              # evaluation
              average_train_loss = running_loss / eval_every
              average_valid_loss = valid_running_loss / len(valid_loader)
              train_loss_list.append(average_train_loss)
              valid_loss_list.append(average_valid_loss)
              global_steps_list.append(global_step)

              # resetting running values
              running_loss = 0.0                
              valid_running_loss = 0.0
              model.train()

        # print progress
        print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, num_epochs, average_train_loss, average_valid_loss))
                
        # checkpoint
        if best_valid_loss > average_valid_loss:
          best_valid_loss = average_valid_loss
          save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
          save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model=model, optimizer=optimizer, num_epochs=10)
