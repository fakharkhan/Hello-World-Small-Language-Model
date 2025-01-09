import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
import os

class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        self.qa_pairs = []
        current_question = None
        current_answer = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Question:'):
                    if current_question and current_answer:
                        self.qa_pairs.append((current_question, current_answer))
                    current_question = line[len('Question:'):].strip()
                    current_answer = None
                elif line.startswith('Answer:'):
                    current_answer = line[len('Answer:'):].strip()
            
            if current_question and current_answer:
                self.qa_pairs.append((current_question, current_answer))
                
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        input_text = f"Question: {question}\nAnswer: {answer}"
        
        tokens = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

class SmallLanguageModel(nn.Module):
    def __init__(self, base_model_name='gpt2'):
        super(SmallLanguageModel, self).__init__()
        self.model = GPT2Model.from_pretrained(base_model_name)
        self.fc = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state)
        return logits

def train_model(epochs=10, batch_size=1, learning_rate=1e-4):
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = QADataset('dataset.txt', tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} QA pairs")

    print("Initializing model...")
    model = SmallLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
    
    print("Training completed. Saving model...")
    torch.save(model.state_dict(), 'qa_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()