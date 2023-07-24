import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

from Training.BlenderBot.Dataset import BlenderbotFinetuneDataset

model_name = 'facebook/blenderbot-400M-distill'
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

# add new special token <k> and </k> to represent the keyword
tokenizer.add_tokens(['<k>', '</k>'])
model.resize_token_embeddings(len(tokenizer.get_vocab()))

dataset = BlenderbotFinetuneDataset(tokenizer=tokenizer)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

epochs = 5
device = 'cuda:0'

model.to(device)
loss_record = []

# Training loop
model.train()
for epoch in range(epochs):
    print("Epoch {}".format(epoch+1))
    progress_bar = tqdm(data_loader, desc="Processing")
    for batch in tqdm(data_loader):
        prompts = batch['prompts']
        response = batch['response']

        input_ids = prompts.to(device)
        labels = response.to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_description('Epoch {} - Loss: {}'.format(epoch, loss))
        loss_record.append(loss)

print(loss_record)

# Save the fine-tuned model
model.save_pretrained("Model/Keyword.pth")




