import argparse
import os
import sys
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class DialogDataset(torch.utils.data.Dataset):
    def __init__(self, full_dialog, previous_utterance_representations, label, tokenizer, max_len, max_utterance):
        self.full_dialog = full_dialog
        self.previous_utterance_representations = previous_utterance_representations
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_utterance = max_utterance

    def __len__(self):
        return len(self.full_dialog)

    def __getitem__(self, item):
        full_dialog = self.full_dialog[item]
        previous_utterance_representations = self.previous_utterance_representations[item]
        label = self.label[item]

        encoding = self.tokenizer.encode_plus(
            full_dialog,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
    
        # Make a tensor of the previous utterance representations. Zero pad if there are less than max_utterance
        if len(previous_utterance_representations) == 0:
            previous_utterance_representations = torch.zeros((self.max_utterance, 768))
        else:
            previous_utterance_representations = torch.stack(previous_utterance_representations)
            if len(previous_utterance_representations) < self.max_utterance:
                previous_utterance_representations = torch.cat(
                    (previous_utterance_representations,
                    torch.zeros((self.max_utterance - len(previous_utterance_representations), 768)))
                )
            else:
                previous_utterance_representations = previous_utterance_representations[:self.max_utterance]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'previous_utterance_representations': previous_utterance_representations,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define a model
class CALIOPERWithoutFinalLayer(nn.Module):
    def __init__(self, bert_model, bert_hidden_size, sentence_bert_hidden_size, dropout):
        super(CALIOPERWithoutFinalLayer, self).__init__()
        self.bert_model = bert_model
        self.bert_hidden_size = bert_hidden_size
        self.sentence_bert_hidden_size = sentence_bert_hidden_size
        self.bert_output_transform = nn.Linear(self.bert_hidden_size, self.sentence_bert_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_hidden_size + sentence_bert_hidden_size, 2)

    def forward(self, input_ids, attention_mask, previous_utterance_representations):
        # Get the [CLS] representation of the full dialog input
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output[1] # [CLS] representation, shape: (batch_size, bert_hidden_size)

        # Trainable layer between the BERT and attention layer
        bert_output_transformed = self.bert_output_transform(bert_output)

        # Get the attention weights
        # bert_output_transformed.unsqueeze(2) -> (batch_size, bert_hidden_size, 1)
        attention = torch.bmm(previous_utterance_representations, bert_output_transformed.unsqueeze(2)).squeeze(2)
        attention = F.softmax(attention, dim=1)

        # Get the weighted sum of the previous utterance representations
        weighted_sum = torch.bmm(previous_utterance_representations.transpose(1, 2), attention.unsqueeze(2)).squeeze(2)

        # Concatenate the [CLS] representation of the full dialog input and the weighted sum
        output = torch.cat((bert_output, weighted_sum), dim=1)

        # Dropout
        output = self.dropout(output)

        # Classifier
        output = self.classifier(output)

        return output

# Define a model
class CALIOPER(nn.Module):
    def __init__(self, bert_model, bert_hidden_size, sentence_bert_hidden_size, dropout):
        super(CALIOPER, self).__init__()
        self.bert_model = bert_model
        self.bert_hidden_size = bert_hidden_size
        self.sentence_bert_hidden_size = sentence_bert_hidden_size
        self.bert_output_transform = nn.Linear(self.bert_hidden_size, self.sentence_bert_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(bert_hidden_size + sentence_bert_hidden_size, bert_hidden_size)
        self.classifier = nn.Linear(bert_hidden_size, 2)

    def forward(self, input_ids, attention_mask, previous_utterance_representations):
        # Get the [CLS] representation of the full dialog input
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output[1] # [CLS] representation, shape: (batch_size, bert_hidden_size)

        # Trainable layer between the BERT and attention layer
        bert_output_transformed = self.bert_output_transform(bert_output)

        # Get the attention weights
        # bert_output_transformed.unsqueeze(2) -> (batch_size, bert_hidden_size, 1)
        attention = torch.bmm(previous_utterance_representations, bert_output_transformed.unsqueeze(2)).squeeze(2)
        attention = F.softmax(attention, dim=1)

        # Get the weighted sum of the previous utterance representations
        weighted_sum = torch.bmm(previous_utterance_representations.transpose(1, 2), attention.unsqueeze(2)).squeeze(2)

        # Concatenate the [CLS] representation of the full dialog input and the weighted sum
        output = torch.cat((bert_output, weighted_sum), dim=1)

        # Dropout
        output = self.dropout(output)

        # Classifier
        output = self.hidden_layer(output)
        output = F.relu(output)
        output = self.classifier(output)

        return output

# Define a model
class CALIOPERWithTrainableAttention(nn.Module):
    # model with layer before query, key, value
    def __init__(self, bert_model, bert_hidden_size, sentence_bert_hidden_size, dropout):
        super(CALIOPERWithTrainableAttention, self).__init__()
        self.bert_model = bert_model
        self.bert_hidden_size = bert_hidden_size
        self.sentence_bert_hidden_size = sentence_bert_hidden_size
        self.bert_output_transform = nn.Linear(self.bert_hidden_size, self.sentence_bert_hidden_size)
        self.key_transform = nn.Linear(self.sentence_bert_hidden_size, self.sentence_bert_hidden_size)
        self.value_transform = nn.Linear(self.sentence_bert_hidden_size, self.sentence_bert_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(bert_hidden_size + sentence_bert_hidden_size, bert_hidden_size)
        self.classifier = nn.Linear(bert_hidden_size, 2)

    def forward(self, input_ids, attention_mask, previous_utterance_representations):
        # Get the [CLS] representation of the full dialog input
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output[1] # [CLS] representation, shape: (batch_size, bert_hidden_size)

        # Trainable layer between the BERT and attention layer
        bert_output_transformed = self.bert_output_transform(bert_output)

        # Get the attention weights
        # bert_output_transformed.unsqueeze(2) -> (batch_size, bert_hidden_size, 1)
        key = self.key_transform(previous_utterance_representations)
        query = bert_output_transformed.unsqueeze(2)
        attention = torch.bmm(key, query).squeeze(2)
        attention = F.softmax(attention, dim=1)

        # Get the weighted sum of the previous utterance representations
        value = self.value_transform(previous_utterance_representations)
        weighted_sum = torch.bmm(value.transpose(1, 2), attention.unsqueeze(2)).squeeze(2)

        # Concatenate the [CLS] representation of the full dialog input and the weighted sum
        output = torch.cat((bert_output, weighted_sum), dim=1)

        # Dropout
        output = self.dropout(output)

        # Classifier
        output = self.hidden_layer(output)
        output = F.relu(output)
        output = self.classifier(output)

        return output
        
# Define a model
class CALIOPERWithMeanpooling(nn.Module):
    # Consider bert_model as a sentence encoder
    # treat the mean of the each token representation as the sentence representation
    def __init__(self, bert_model, bert_hidden_size, sentence_bert_hidden_size, dropout):
        super(CALIOPERWithMeanpooling, self).__init__()
        self.bert_model = bert_model
        self.bert_hidden_size = bert_hidden_size
        self.sentence_bert_hidden_size = sentence_bert_hidden_size
        self.bert_output_transform = nn.Linear(self.bert_hidden_size, self.sentence_bert_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(bert_hidden_size + sentence_bert_hidden_size, bert_hidden_size)
        self.classifier = nn.Linear(bert_hidden_size, 2)

    def forward(self, input_ids, attention_mask, previous_utterance_representations):
        # Get the mean representation of the input tokens
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_output[0] # token representations, shape: (batch_size, sequence_length, bert_hidden_size)
        mean_pooled = token_embeddings.sum(axis=1) / attention_mask.sum(axis=1).unsqueeze(1) # shape: (batch_size, bert_hidden_size)

        # Trainable layer between the BERT and attention layer
        bert_output_transformed = self.bert_output_transform(mean_pooled)

        # Get the attention weights
        # bert_output_transformed.unsqueeze(2) -> (batch_size, bert_hidden_size, 1)
        attention = torch.bmm(previous_utterance_representations, bert_output_transformed.unsqueeze(2)).squeeze(2)
        attention = F.softmax(attention, dim=1)

        # Get the weighted sum of the previous utterance representations
        weighted_sum = torch.bmm(previous_utterance_representations.transpose(1, 2), attention.unsqueeze(2)).squeeze(2)

        # Concatenate the [CLS] representation of the full dialog input and the weighted sum
        output = torch.cat((mean_pooled, weighted_sum), dim=1)

        # Dropout
        output = self.dropout(output)

        # Classifier
        output = self.hidden_layer(output)
        output = F.relu(output)
        output = self.classifier(output)

        return output

# Define a model
class DialogFinetuneModel(nn.Module):
    def __init__(self, bert_model, bert_hidden_size, sentence_bert_hidden_size, dropout):
        super(DialogFinetuneModel, self).__init__()
        self.bert_model = bert_model
        self.bert_hidden_size = bert_hidden_size
        #self.sentence_bert_hidden_size = sentence_bert_hidden_size
        #self.bert_output_transform = nn.Linear(self.bert_hidden_size, self.sentence_bert_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.classifier = nn.Linear(bert_hidden_size, 2)

    def forward(self, input_ids, attention_mask, previous_utterance_representations):
        # Get the mean representation of the input tokens
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output[1] # [CLS] representation, shape: (batch_size, bert_hidden_size)
        
        # Trainable layer between the BERT and attention layer
        #bert_output_transformed = self.bert_output_transform(bert_output)

        # Get the attention weights
        # bert_output_transformed.unsqueeze(2) -> (batch_size, bert_hidden_size, 1)
        #attention = torch.bmm(previous_utterance_representations, bert_output_transformed.unsqueeze(2)).squeeze(2)
        #attention = F.softmax(attention, dim=1)

        # Get the weighted sum of the previous utterance representations
        #weighted_sum = torch.bmm(previous_utterance_representations.transpose(1, 2), attention.unsqueeze(2)).squeeze(2)

        output = bert_output

        # Dropout
        output = self.dropout(output)

        # Classifier
        output = self.hidden_layer(output)
        output = F.relu(output)
        output = self.classifier(output)

        return output


# Define a model
class DialogFinetuneModelMeanpooling(nn.Module):
    def __init__(self, bert_model, bert_hidden_size, sentence_bert_hidden_size, dropout):
        super(DialogFinetuneModelMeanpooling, self).__init__()
        self.bert_model = bert_model
        self.bert_hidden_size = bert_hidden_size
        #self.sentence_bert_hidden_size = sentence_bert_hidden_size
        #self.bert_output_transform = nn.Linear(self.bert_hidden_size, self.sentence_bert_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.classifier = nn.Linear(bert_hidden_size, 2)

    def forward(self, input_ids, attention_mask, previous_utterance_representations):
        # Get the mean representation of the input tokens
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_output[0] # token representations, shape: (batch_size, sequence_length, bert_hidden_size)
        mean_pooled = token_embeddings.sum(axis=1) / attention_mask.sum(axis=1).unsqueeze(1) # shape: (batch_size, bert_hidden_size)
        
        # Trainable layer between the BERT and attention layer
        #bert_output_transformed = self.bert_output_transform(mean_pooled)

        # Get the attention weights
        # bert_output_transformed.unsqueeze(2) -> (batch_size, bert_hidden_size, 1)
        #attention = torch.bmm(previous_utterance_representations, bert_output_transformed.unsqueeze(2)).squeeze(2)
        #attention = F.softmax(attention, dim=1)

        # Get the weighted sum of the previous utterance representations
        #weighted_sum = torch.bmm(previous_utterance_representations.transpose(1, 2), attention.unsqueeze(2)).squeeze(2)

        # Concatenate the [CLS] representation of the full dialog input and the weighted sum
        output = mean_pooled

        # Dropout
        output = self.dropout(output)

        # Classifier
        output = self.hidden_layer(output)
        output = F.relu(output)
        output = self.classifier(output)

        return output

# argparse
def parse_args():
    '''
    Parse input arguments

    Returns:
        args: parsed input arguments
    '''
    parser = argparse.ArgumentParser(description='Dialog Retrieval')
    # Train data. One of simsimi, fair
    parser.add_argument('--train_data', dest='train_data', help='Training data', default='simsimi', type=str)
    # Test data. One of simsimi, fair. If not specified, the same as train_data
    parser.add_argument('--test_data', dest='test_data', help='Test data', default=None, type=str)
    # Speaker tags. If A, use alternating speaker tags. If None, don't use speaker tags
    parser.add_argument('--speaker_tags', dest='speaker_tags', help='Speaker tags', default='A', type=str)
    # Utterance seperator token. If None, don't use utterance seperator token
    parser.add_argument('--utterance_seperator', dest='utterance_seperator', help='Utterance seperator token', default='[SEP]', type=str)
    # Main encoder name. From huggingface
    parser.add_argument('--main_encoder', dest='main_encoder', help='Main encoder name', default='bert-base-uncased', type=str)
    # Context encoder name. From huggingface
    parser.add_argument('--context_encoder', dest='context_encoder', help='Context encoder name', default='bert-base-uncased', type=str)
    # Whether to use full dialog or only the last utterance
    parser.add_argument('--use_only_last', dest='use_only_last', help='Whether to use only the last utterance', action='store_true')
    # Whether to use mean pooling
    parser.add_argument('--mean_pooling', dest='mean_pooling', help='Whether to use mean pooling', action='store_true')
    # Whether to use attention
    parser.add_argument('--no_attention', dest='no_attention', help='Whether to use attention', action='store_true')
    # Use model with trainable layer before attention
    parser.add_argument('--trainable_attention', dest='trainable_attention', help='Use model with trainable layer before attention', action='store_true')
    # Max context utterances
    parser.add_argument('--max_context_utterances', dest='max_context_utterances', help='Max context utterances', default=10, type=int)
    # Epochs
    parser.add_argument('--epochs', dest='epochs', help='Epochs', default=10, type=int)
    # Batch size
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size', default=32, type=int)
    # GPUs to use
    parser.add_argument('--gpus', dest='gpus', help='GPUs to use', default='0', type=str)
    # repeat
    parser.add_argument('--repeat', dest='repeat', help='Repeat', default=1, type=int)
    
    args = parser.parse_args()
    return args

def load_data(name, split):
    '''
    Load data

    Args:
        name: dataset name
        split: train or test

    Returns:
        data: loaded data
            data['text']: list of target utterances
            data['label']: list of labels
            data['all_utterances']: list of utterances
    '''
    if name == 'simsimi':
        data = load_simsimi(split)
    elif name == 'fair':
        pass
    else:
        raise ValueError('Invalid dataset name')
    
    data_dict = {}
    data_dict['text'] = data['text'].tolist()
    data_dict['label'] = data['label'].tolist()
    data_dict['all_utterances'] = data['all_utterances'].tolist()

    return data_dict

def load_simsimi(split):
    '''
    Load SimSimi dataset

    Args:
        split: train or test

    Returns:
        data: Pandas dataframe
            data['text']: list of target utterances
            data['label']: list of labels
            data['all_utterances']: list of utterances
    '''
    if split == 'train':
        file_path = 'data/train_not_U.tsv'
    elif split == 'test':
        file_path = 'data/test_not_U.tsv'
    else:
        raise ValueError('Invalid split')
    
    data = pd.read_csv(file_path, sep='\t', index_col=0)
    data['all_utterances'] = data['previous_utterance'].apply(lambda x: eval(x))
    data['label'] = data['offensive'].apply(lambda x: 1 if x == 'Y' else 0)

    return data

def load_fair(split):
    '''
    Load FAIR dataset

    Args:
        split: train or test

    Returns:
        data: Pandas dataframe
            data['text']: list of target utterances
            data['label']: list of labels
            data['all_utterances']: list of utterances
    '''
    if split == 'train':
        file_path = 'data/multi_turn_safety_train.csv'
    elif split == 'test':
        file_path = 'data/multi_turn_safety_test.csv'
    elif split == 'valid':
        file_path = 'data/multi_turn_safety_valid.csv'
    else:
        raise ValueError('Invalid split')
    
    data = pd.read_csv(file_path)
    data['all_utterances'] = data['previous_utterance'].apply(lambda x: eval(x))
    data['label'] = data['offensive'].apply(lambda x: 1 if x == 'Y' else 0)

    return data

def train(model, dataloader, loss_fn, optimizer, scheduler, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        previous_utterance_representations = batch['previous_utterance_representations'].to(device)
        labels = batch['label'].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            previous_utterance_representations=previous_utterance_representations
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def evaluate(model, dataloader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            previous_utterance_representations = batch['previous_utterance_representations'].to(device)
            labels = batch['label'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                previous_utterance_representations=previous_utterance_representations
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)

def main():
    args = parse_args()

    # Set the GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    if args.train_data == 'simsimi':
        train_data = load_data('simsimi', 'train')
    elif args.train_data == 'fair':
        train_data = load_data('fair', 'train')
    else:
        raise ValueError('Invalid train data')
    
    dep_test_data_index = None
    if args.test_data is None:
        test_data = load_data(args.train_data, 'test')
    else:
        if args.test_data == 'simsimi':
            test_data = load_data('simsimi', 'test')
            data_simsimi_test = load_simsimi('test')
            dep_test_data_index = (data_simsimi_test.context_dependent == 'Y') | (data_simsimi_test.offensive == 'N')
        elif args.test_data == 'fair':
            test_data = load_data('fair', 'test')
        else:
            raise ValueError('Invalid test data')
        
    # Get train data as lists
    train_text = train_data['text']
    train_label = train_data['label']
    train_all_utterances = train_data['all_utterances']
    train_previous_utterance = [None] * len(train_all_utterances)

    assert len(train_text) == len(train_label) == len(train_all_utterances)

    if args.speaker_tags == 'A':
        speaker_tags = ['A: ', 'B: ']
    elif args.speaker_tags is None:
        speaker_tags = ['']

    for i in range(len(train_all_utterances)):
        # Remove the last utterance to make previous_utterances
        train_previous_utterance[i] = train_all_utterances[i][:-1]

        # Append speaker tags to the previous utterances
        for j in range(len(train_previous_utterance[i])):
            speaker_tag = speaker_tags[(len(train_previous_utterance[i]) - j) % 2]
            train_previous_utterance[i][j] = speaker_tag + train_previous_utterance[i][j]

        # Append speaker tag to the target utterance
        train_text[i] = speaker_tags[0] + train_text[i]
    
    train_full_dialog = []
    for i in range(len(train_text)):
        dialog = ''
        for j in range(len(train_previous_utterance[i])):
            dialog += train_previous_utterance[i][j] + args.utterance_seperator
        dialog += train_text[i]
        train_full_dialog.append(dialog)

    # Get test data as lists
    test_text = test_data['text']
    test_label = test_data['label']
    test_all_utterances = test_data['all_utterances']
    test_previous_utterance = [None] * len(test_all_utterances)

    assert len(test_text) == len(test_label) == len(test_all_utterances)

    for i in range(len(test_all_utterances)):
        # Remove the last utterance to make previous_utterances
        test_previous_utterance[i] = test_all_utterances[i][:-1]

        # Append speaker tags to the previous utterances
        for j in range(len(test_previous_utterance[i])):
            speaker_tag = speaker_tags[(len(test_previous_utterance[i]) - j) % 2]
            test_previous_utterance[i][j] = speaker_tag + test_previous_utterance[i][j]

        # Append speaker tag to the target utterance
        test_text[i] = speaker_tags[0] + test_text[i]

    test_full_dialog = []
    for i in range(len(test_text)):
        dialog = ''
        for j in range(len(test_previous_utterance[i])):
            dialog += test_previous_utterance[i][j] + args.utterance_seperator
        dialog += test_text[i]
        test_full_dialog.append(dialog)

    # Load Sentence-BERT model
    context_encoder_name = args.context_encoder
    context_encoder_name_short = context_encoder_name.split('/')[-1]

    sentence_bert = SentenceTransformer(context_encoder_name_short)

    if os.path.exists(f'./models/sentence_bert_dict_{context_encoder_name_short}.pkl'):
        sentence_bert_dict = pickle.load(open(f'./models/sentence_bert_dict_{context_encoder_name_short}.pkl', 'rb'))
    else:
        sentence_bert_dict = {}
    # Get sentence embeddings of each utterance in the train data
    train_previous_utterance_representations = []
    for i in tqdm(range(len(train_previous_utterance))):
        previous_utterance_representations = []
        for j in range(len(train_previous_utterance[i])):
            if train_previous_utterance[i][j] in sentence_bert_dict:
                previous_utterance_representations.append(sentence_bert_dict[train_previous_utterance[i][j]])
            else:
                previous_utterance_representations.append(sentence_bert.encode(train_previous_utterance[i][j]))
                sentence_bert_dict[train_previous_utterance[i][j]] = sentence_bert.encode(train_previous_utterance[i][j])
        train_previous_utterance_representations.append(previous_utterance_representations)
    # To Tensor
    for i in range(len(train_previous_utterance_representations)):
        for j in range(len(train_previous_utterance_representations[i])):
            train_previous_utterance_representations[i][j] = torch.tensor(train_previous_utterance_representations[i][j])

    # Get sentence embeddings of each utterance in the test data
    test_previous_utterance_representations = []
    for i in tqdm(range(len(test_previous_utterance))):
        previous_utterance_representations = []
        for j in range(len(test_previous_utterance[i])):
            if test_previous_utterance[i][j] in sentence_bert_dict:
                previous_utterance_representations.append(sentence_bert_dict[test_previous_utterance[i][j]])
            else:
                previous_utterance_representations.append(sentence_bert.encode(test_previous_utterance[i][j]))
                sentence_bert_dict[test_previous_utterance[i][j]] = sentence_bert.encode(test_previous_utterance[i][j])
        test_previous_utterance_representations.append(previous_utterance_representations)
    # To Tensor
    for i in range(len(test_previous_utterance_representations)):
        for j in range(len(test_previous_utterance_representations[i])):
            test_previous_utterance_representations[i][j] = torch.tensor(test_previous_utterance_representations[i][j])

    # Save sentence_bert_dict
    with open(f'./models/sentence_bert_dict_{context_encoder_name_short}.pkl', 'wb') as f:
        pickle.dump(sentence_bert_dict, f)

    # Load tokenizer
    main_encoder_name = args.main_encoder
    main_encoder_name_short = main_encoder_name.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(main_encoder_name)

    train_dataset = DialogDataset(
        full_dialog=train_full_dialog if not args.use_only_last else train_text,
        previous_utterance_representations=train_previous_utterance_representations,
        label=train_label,
        max_len=512,
        max_utterance=args.max_context_utterances,
        tokenizer=tokenizer
    )

    test_dataset = DialogDataset(
        full_dialog=test_full_dialog if not args.use_only_last else test_text,
        previous_utterance_representations=test_previous_utterance_representations,
        label=test_label,
        max_len=512,
        max_utterance=args.max_context_utterances,
        tokenizer=tokenizer
    )

    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=4
    )

    bert = AutoModel.from_pretrained(main_encoder_name)

    # evaluation results dataframe
    eval_results = pd.DataFrame(columns=['model', 'accuracy', 'f1', 'dep_f1'])

    for repeat_count in range(args.repeat):
        if args.trainable_attention:
            model = CALIOPERWithTrainableAttention(bert, 768, 768, dropout=0.1)
        elif args.no_attention:
            if args.mean_pooling:
                model = DialogFinetuneModelMeanpooling(bert, 768, 768, dropout=0.1)
            else:
                model = DialogFinetuneModel(bert, 768, 768, dropout=0.1)
        else:
            if args.mean_pooling:
                model = CALIOPERWithMeanpooling(bert, 768, 768, dropout=0.1)
            else:
                model = CALIOPER(bert, 768, 768, dropout=0.1)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5)

        epochs = args.epochs
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss().to(device)

        # Train the model
        history = defaultdict(list)
        best_accuracy = 0
        for epoch in tqdm(range(epochs)):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            train_acc, train_loss = train(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                scheduler,
                device,
                len(train_dataset)
            )
            print(f'Train loss {train_loss} accuracy {train_acc}')
            test_acc, test_loss = evaluate(
                model,
                test_dataloader,
                loss_fn,
                device,
                len(test_dataset)
            )
            print(f'Test loss {test_loss} accuracy {test_acc}')
            print()
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['test_acc'].append(test_acc)
            history['test_loss'].append(test_loss)

        y_true = []
        y_pred = []
        y_pred_prob = []
        model.eval()
        for d in tqdm(test_dataloader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)
            previous_utterance_representations = d['previous_utterance_representations'].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    previous_utterance_representations=previous_utterance_representations
                )

            _, preds = torch.max(outputs, dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_pred_prob.extend(np.exp(outputs[:, 1].detach().cpu().numpy()) / (np.exp(outputs[:, 1].detach().cpu().numpy()) + np.exp(outputs[:, 0].detach().cpu().numpy())))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_prob = np.array(y_pred_prob)

        print(f"F1-score: {f1_score(y_true, y_pred)}")
        if dep_test_data_index is not None:
            print(f"Dep-F1-score: {f1_score(y_true[dep_test_data_index], y_pred[dep_test_data_index])}")

        # save model
        # model/dialog_retrieval_model_{train_data}_{main_encoder_name_short}_{context_encoder_name_short}_{nocontext|withcontext}_{timestamp}.pt
        use_context = 'withcontext' if not args.use_only_last else 'nocontext'
        use_mean_pooling = '_meanpooling' if args.mean_pooling else ''
        use_attention = '_noattention' if args.no_attention else ''
        use_trainable_attention = '_trainableattention' if args.trainable_attention else ''
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_suffix = f'{main_encoder_name_short}_{context_encoder_name_short}_{use_context}{use_mean_pooling}{use_attention}{use_trainable_attention}_{timestamp}'
        model_filename = f'models/dialog_retrieval_model_{args.train_data}_{model_suffix}.pt'
        torch.save(model.state_dict(), model_filename)

        # save evaluation results
        eval_results = eval_results.append({
            'model': model_filename,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'dep_f1': f1_score(y_true[dep_test_data_index], y_pred[dep_test_data_index]) if dep_test_data_index is not None else None
        }, ignore_index=True)

    # save evaluation results
    eval_results.to_csv(f'results/eval_results_{args.train_data}_{args.test_data}_{model_suffix}.csv', index=False)

    # print evaluation results
    print(eval_results)

    print(eval_results.mean())

if __name__ == '__main__':
    main()