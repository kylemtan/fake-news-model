import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import random

# Helper functions
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Seed PyTorch for hyperparameter tweaking
seed_val = 21

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

# Set up device for GPU if available, otherwise CPU
if torch.mps.is_available():
    device = torch.device("mps")
    print('There are %d GPU(s) available. Using GPU.' % torch.mps.device_count())
else:
    print("No GPU is available. Using CPU.")
    device = torch.device("cpu")

# Load the dataset into a pandas dataframe.
fake_df = pd.read_csv("../data/Fake.csv", delimiter=',', header=None, names=['title', 'text', 'subject', 'date'])
real_df = pd.read_csv("../data/True.csv", delimiter=',', header=None, names=['title', 'text', 'subject', 'date'])

fake_df['label'] = 1
real_df['label'] = 0

df = pd.concat([fake_df, real_df])

df["text"]= df["title"] + " " + df["text"]

df = df[["text","label"]]

print(df.sample(10))

# Take the cased BERT version since capitalization could matter for real or fake articles - check to switch
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Extract titles and text and tokenize
texts = df.text.values
labels = df.label.values

# Tokenize all of the sentences and map the tokens to their word IDs.
input_ids = []
attention_masks = []

for text in texts:
    encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens = True,
                        max_length = 100,
                        truncation=True,
                        padding = 'max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

print('Original: ', texts[0])
print('Token IDs:', input_ids[0])

# Split the dataset 80-20
dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Hyperparameter to be tweaked: 16 or 32
batch_size = 16

train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

if torch.mps.is_available():
    model.to(device)
    print('Model running on GPU')
else:
    model.cpu()
    print('Model running on CPU')

# Tweakable hyperparameters:
#   Learning rate (Adam): 5e-5, 3e-5, 2e-5
#   eps: 1e-08 
#   Number of epochs: 2, 3, 4
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-08)
epochs = 2

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

training_stats = []

total_t0 = time.time()

# Training the model:
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        result = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels,
                       return_dict=True)

        loss = result.loss
        logits = result.logits

        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Training Time': training_time,
        }
    )

print("")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
print("")
print('Predicting labels...')

model.eval()

predictions , true_labels = [], []

# Using the model for prediction:
for batch in test_dataloader:
  batch = tuple(t.to(device) for t in batch)

  b_input_ids, b_input_mask, b_labels = batch

  with torch.no_grad():
      result = model(b_input_ids,
                     token_type_ids=None,
                     attention_mask=b_input_mask,
                     return_dict=True)

  logits = result.logits

  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  predictions.append(logits)
  true_labels.append(label_ids)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = np.concatenate(true_labels, axis=0)

correct_predictions = 0

for i in range(len(flat_predictions)):
    if (flat_predictions[i] == flat_true_labels[i]):
        correct_predictions += 1

print("Accuracy: ", correct_predictions / len(flat_true_labels))