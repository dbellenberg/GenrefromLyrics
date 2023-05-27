import torch
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr
from transformers import BertTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchmetrics
import re


class LyricsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {'input_ids': torch.as_tensor(self.encodings.iloc[idx])}
        item['labels'] = torch.as_tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.encodings)

class LyricsClassifier(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased', num_labels=5): 
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertForSequenceClassification.from_pretrained(self.hparams.model_name,
                                                                  num_labels=self.hparams.num_labels)
        self.accuracy = torchmetrics.Accuracy(task="multiclass",compute_on_step=False, num_classes=num_labels)

    def forward(self, input_ids, labels=None):
        return self.bert(input_ids, labels=labels)
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['labels'])
        _, predicted = torch.max(outputs.logits, 1)
        correct = (predicted == batch['labels']).sum().item()
        accuracy = correct / len(batch['labels'])
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return accuracy
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)

model = LyricsClassifier.load_from_checkpoint(checkpoint_path="epoch=1-step=145040.ckpt", map_location=torch.device('cpu'))

def strip_lyrics(lyrics):
    # Remove strings enclosed in brackets []
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    
    # Remove substrings starting with a backslash \
    lyrics = re.sub(r'\\[^\s]*', '', lyrics)

    # Remove newline characters \n
    lyrics = re.sub(r'\n', ' ', lyrics)
    
    # Remove single quotes '
    lyrics = re.sub(r"'", '', lyrics)
    
    # Remove leading and trailing whitespaces
    lyrics = lyrics.strip()

    # Strip the string and ensure only one space between words
    lyrics = re.sub(r'\s+', ' ', lyrics.strip())

    return lyrics

def predict_genre(Artist, Title, Lyrics):
    lyrics = strip_lyrics(Lyrics)  # Preprocess the lyrics
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(lyrics, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    probabilities = F.softmax(outputs.logits, dim=-1)
    probabilities = probabilities.cpu().numpy()

    label_map = {0: 'Country', 1: 'Pop', 2: 'Rap', 3: 'R&B', 4: 'Rock'}
    
    probabilities_dict = {label_map[i]: float(prob) for i, prob in enumerate(probabilities[0])}  # convert numpy.float32 to float

    return probabilities_dict


#description = '<img src="https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_CMYK_Green.png" alt="Spotify Logo">'
description = '<img src="https://i.imgur.com/q4xD7ry.png" alt="BertBeats Logo">'

iface = gr.Interface(
    fn=predict_genre,
    inputs=[
        gr.inputs.Textbox(lines=1, placeholder='Artist Here...'),
        gr.inputs.Textbox(lines=1, placeholder='Title Here...'),
        gr.inputs.Textbox(lines=4, placeholder='Lyrics Here...'),
        #gr.inputs.File()
    ],
    outputs=gr.outputs.Label(num_top_classes=5, label="Genre Suggestion"),
    description=description
)
iface.launch()



