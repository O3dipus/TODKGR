import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from Knowledge.ConceptNet import ConceptNet


class TGConvDataset:
    def __init__(self, conceptnet, sentence_encoder):
        self.train_path = 'Data/tgconv_train.json'
        self.test_path = 'Data/tgconv_test.json'
        self.sentence_encoder = sentence_encoder
        self.conceptnet = conceptnet

        self.train_data = []
        with open(self.train_path) as f:
            lines = f.readlines()
            self.train_data = [eval(line.strip()) for line in lines]

        self.test_data = []
        with open(self.test_path) as f:
            lines = f.readlines()
            self.test_data = [eval(line.strip()) for line in lines]

    def extract_context_keyword(self):
        episodes = []
        for data in tqdm(self.train_data):
            dialog = data['dialog']
            dialog_length = len(dialog)
            entity_path = data['entity_path']
            target = torch.tensor([self.conceptnet.indexOf(entity_path[-1])]).cuda()

            episode = []
            for i in range(1, dialog_length-1):
                context = ''.join(dialog[:i-1])
                response = dialog[i]
                keyword = entity_path[i+1]
                action = self.conceptnet.indexOf(keyword)

                encoded = self.sentence_encoder.encode([context, response], convert_to_tensor=True)
                observation = torch.cat((target, encoded[0], encoded[1]))
                episode.append((observation, action))
            episodes.append(episode)
        return episode



