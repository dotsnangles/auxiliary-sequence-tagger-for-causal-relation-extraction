import torch
import torch.nn as nn

from transformers import ElectraModel, ElectraTokenizer

class BERTPoSTagger(nn.Module):
    def __init__(self,
                 bert,
                 output_dim, 
                 dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text, mask):
        #text = [batch size, sent len]
        embedded = self.dropout(self.bert(text, mask)[0])
        #embedded = [batch size, seq len, emb dim]
        predictions = self.fc(self.dropout(embedded))
        return predictions
    
bert = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained('./tokenizer/')
# print(len(tokenizer))

label2id = {'[PAD]': 0, '-': 1, 'AFW_B': 2, 'AFW_I': 3, 'ANM_B': 4, 'ANM_I': 5, 'CVL_B': 6, 'CVL_I': 7, 'DAT_B': 8, 'DAT_I': 9, 'LOC_B': 10, 'LOC_I': 11, 'MAT_B': 12, 'MAT_I': 13, 'NUM_B': 14, 'NUM_I': 15, 'TIM_B': 16, 'TIM_I': 17, 'TRM_B': 18, 'TRM_I': 19, 'WRK_B': 20, 'WRK_I': 21}
id2label = {0: '[PAD]', 1: '-', 2: 'AFW_B', 3: 'AFW_I', 4: 'ANM_B', 5: 'ANM_I', 6: 'CVL_B', 7: 'CVL_I', 8: 'DAT_B', 9: 'DAT_I', 10: 'LOC_B', 11: 'LOC_I', 12: 'MAT_B', 13: 'MAT_I', 14: 'NUM_B', 15: 'NUM_I', 16: 'TIM_B', 17: 'TIM_I', 18: 'TRM_B', 19: 'TRM_I', 20: 'WRK_B', 21: 'WRK_I'}
labels = ['[PAD]', '-', 'AFW_B', 'AFW_I', 'ANM_B', 'ANM_I', 'CVL_B', 'CVL_I', 'DAT_B', 'DAT_I', 'LOC_B', 'LOC_I', 'MAT_B', 'MAT_I', 'NUM_B', 'NUM_I', 'TIM_B', 'TIM_I', 'TRM_B', 'TRM_I', 'WRK_B', 'WRK_I']

OUTPUT_DIM = 22 # len(labels)
DROPOUT = 0.25
model = BERTPoSTagger(bert, OUTPUT_DIM, DROPOUT)
model.bert.resize_token_embeddings(len(tokenizer))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NGPU = torch.cuda.device_count()
if NGPU > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(NGPU)))
    # torch.multiprocessing.set_start_method('spawn', force=True)
model = model.to(device)

model.load_state_dict(torch.load("tut2-model.pt"))