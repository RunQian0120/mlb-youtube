import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from process_data import create_all_data, all_labels
import time
from tqdm import tqdm
import nltk
from collections import Counter

nltk.download('punkt_tab')

def tokenize_caption(caption):
    return nltk.word_tokenize(caption.lower())

def build_vocab(captions, vocab_size=5000):
    word_counts = Counter(word for caption in captions for word in tokenize_caption(caption))
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(vocab_size), start=1)}
    vocab["<unk>"] = 0
    return vocab

class VideoCNN(nn.Module):
    def __init__(self):
        super(VideoCNN, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3d_2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3d_3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.global_avg_pool = nn.AdaptiveAvgPool3d((4, 4, 4))  # Ensures a fixed-size output
        self.fc = nn.Linear(64 * 4 * 4 * 4, 256)  # Adjust shape based on input size
    
    def forward(self, x):
        x = F.relu(self.conv3d_1(x))
        x = self.pool(x)
        x = F.relu(self.conv3d_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3d_3(x))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class CaptionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(CaptionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 256)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x

class VideoCaptionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, num_classes=3):
        super(VideoCaptionClassifier, self).__init__()
        self.video_model = VideoCNN()
        self.caption_model = CaptionLSTM(vocab_size, embed_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, video, caption):
        video_features = self.video_model(video)
        caption_features = self.caption_model(caption)
        combined = torch.cat((video_features, caption_features), dim=1)
        output = self.fc(combined)
        return output

class BaseballDataset(Dataset):
    def __init__(self, data, vocab, transform=None):
        self.data = data
        self.transform = transform
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        images = [img.convert('RGB') for img in self.data[idx]['images']]
        if self.transform:
            images = [self.transform(img) for img in images]
        video_tensor = torch.stack(images)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        caption_tokens = tokenize_caption(self.data[idx]['caption'])
        caption_indices = [self.vocab.get(token, self.vocab["<unk>"]) for token in caption_tokens]
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)
        
        label = self.data[idx]['label']
        return video_tensor, caption_tensor, torch.tensor(label, dtype=torch.long)

all_data = create_all_data()
captions = [item["caption"] for item in all_data]
vocab = build_vocab(captions)

# Training and Testing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = BaseballDataset(all_data, vocab, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: (torch.stack([x[0] for x in batch]), nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True), torch.tensor([x[2] for x in batch])))

model = VideoCaptionClassifier(vocab_size=len(vocab), num_classes=len(all_labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    losses = []
    accus = []
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)
        
        for video, caption, label in progress_bar:
            optimizer.zero_grad()
            output = model(video, caption)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == label).sum().item()
            total += label.size(0)
            
            progress_bar.set_postfix(loss=loss.item())
        
        accuracy = correct / total * 100
        epoch_time = time.time() - start_time
        losses.append(total_loss / len(dataloader))
        accus.append(accuracy)
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Loss: {total_loss / len(dataloader)}, Accuracy: {accuracy:.2f}%")

    print(losses)
    print(accus)


train(model, dataloader, criterion, optimizer)
