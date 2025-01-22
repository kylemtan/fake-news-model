Kyle Macasilli-Tan
macasill@usc.edu

This project uses transfer learning on the BERT model to separate real news from fake news.

# Dataset

This project uses the "Fake News Detection Datasets" dataset from Kaggle. The separate real and fake CSV files were combined, 
and given labels 1 for fake and 0 for real. The title and text of the article were then combined into one string while the subject
and date of the article were removed, since both of these features were found to have no correlation to whether or not the article
was real or fake. The combined articles were then truncated to a max length of 100 due to computational limitations. This resulted in a final dataset of ~45,000 unique title and text strings labelled as real or fake.

# Model Development and Training

In deciding which BERT model would be most adequate, the bert-base-cased model was first used due to to the notion that capitalization
in title as hooks for readers might be an indication of fake news and simply due to the sheer amount of proper nouns used in news
articles. The model was eventually run again with the bert-base-uncased model under the same hyperparameters and this hypothesis was 
confirmed due to a slightly lower but statistically significant difference in accuracy. 

The hyperparameters that were tested were batch size, train and test split (after validation), learning rate and epsilon for AdamW, and the number of epochs. The tested ranges for each of these hyperparameters came from the suggestions provided by the creators of the BERT model to ensure a good fit for the task.

# Model Evaluation/Results

Accuracy was chosen as the key metric since the dataset was evenly split between the two binary categories. After running evaluation
with the found ideal hyperparameters and model type, the best performing combination resulted in an accuracy of 0.9997 and an average
loss of 0.01.

# Discussion

The dataset used was good for the task at hand, but future attempts at the same task might benefit from looking into using datasets where the labelled real news is from multiple sources rather than just one website (Reuters.com) as well as a dataset more balanced towards multiple kinds of articles rather than just political news. 

The BERT model architecture is ideal for the task of detecting fake news since it is able to understand bidirectional context and link words and phrases in large pieces of text like news articles. Additionally, its extensive pre-training allows it to be effective in identifying common speech patterns that indicate deception or untrustworthiness. Similarly, the training procedure is based off of the standard set by the BERT model creators for the same reasons and can also be considered a good fit.

Accuracy was a representative metric for the balanced dataset and also applicable to the models possible application as accuracy is the
most important factor.

This model could possibly be used to determine fake text in general beyond fake news; however, more extensive training and testing would need to be done before conclusively making such an assertion. The model certainly contributes to social good as it can be used to 
ensure that information from news articles is authentic and not misguiding, keeping the public safe from news with malicious intent.

Future steps for this project might include using it on datasets with a more diverse and overall representative dataset to ensure its 
reliability over multiple news outlets and news categories. Training it on a more computationally powerful system would also allow for 
the entire article to be processed and likely lead to even better results. Finally, this model could be made available for identifying fake news by deploying it as an API.

# Sources

PyTorch documentation at https://pytorch.org/docs/stable/index.html
BERT documentation at https://huggingface.co/docs/transformers/en/model_doc/bert
Lesson 4/5 Notebook in Google Colab
General Resources listed in CAIS++ Curriculum Headquarters