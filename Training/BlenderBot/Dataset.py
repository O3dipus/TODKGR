from torch.utils.data import Dataset


class BlenderbotFinetuneDataset(Dataset):
    def __init__(self, tokenizer):
        self.dataset = None
        self.tokenizer = tokenizer
        self.load_data('../Data/tgconv_train.json')

    @staticmethod
    def special_tokens(keywords, contexts):
        keyword_tokens = '<k>' + ' '.join(keywords) + '</k>'
        context_tokens = '<s>' + '</s><s>'.join(contexts) + '</s>'
        return keyword_tokens + context_tokens

    def load_data(self, path):
        f = open(path)
        lines = f.readlines()

        dataset = []
        for line in lines:
            data = eval(line.strip())
            dialogs = data['dialog']
            concepts = data['concepts']

            for idx in range(1, len(dialogs)-1):
                contexts = dialogs[:idx]
                response = dialogs[idx]
                keywords = concepts[idx-1]

                keyword_tokens = '<k>' + ' '.join(keywords) + '</k>'
                context_tokens = '<s>' + '</s><s>'.join(contexts) + '</s>'
                concat_tokens = keyword_tokens + context_tokens

                dataset.append({'prompts': concat_tokens, 'response': response})
        self.dataset = dataset
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dialogue = self.dataset[idx]
        input_text = dialogue['prompts']
        response_text = dialogue['response']

        # Tokenize input and response texts
        input_tokens = self.tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True,
                                      max_length=128)
        response_tokens = self.tokenizer(response_text, return_tensors='pt', padding='max_length', truncation=True,
                                         max_length=128)

        return {
            'prompts': input_tokens['input_ids'].squeeze(),
            'attention_mask': input_tokens['attention_mask'].squeeze(),
            'response': response_tokens['input_ids'].squeeze(),
        }
