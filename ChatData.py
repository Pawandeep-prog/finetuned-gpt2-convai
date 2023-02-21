from torch.utils.data import Dataset
import json

class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))

        self.X = []
        for i in self.data:
            for j in i['dialog']:
                self.X.append(j['text'])

        for idx, i in enumerate(self.X):
            try:
                self.X[idx] = "<startofstring> "+i+" <bot>: "+self.X[idx+1]+" <endofstring>"
            except:
                break

        self.X = self.X[:5000]
        
        print(self.X[0])

        self.X_encoded = tokenizer(self.X,max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])