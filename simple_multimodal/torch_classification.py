import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
from process_data import create_all_data, all_labels, create_test_data
import time
from tqdm import tqdm
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score 
import os

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
        self.convs = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) 
            for _ in range(4)
        ])
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16*4*32*32, 256)

    def forward(self, x):
        per_frame_features = []
        for i, conv in enumerate(self.convs):
            frame_i = x[:, :, i, :, :]
            out_i = F.relu(conv(frame_i))
            out_i = self.pool(out_i)
            per_frame_features.append(out_i)
        
        merged = torch.cat(per_frame_features, dim=1)
        merged = merged.view(merged.size(0), -1)
        merged = self.fc(merged)
        return merged

    def get_weight_norms(self):
        norms = []
        for conv in self.convs:
            norms.append(conv.weight.norm().item())
        return norms

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
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, num_classes=3, dropout_rate=0.4):
        super(VideoCaptionClassifier, self).__init__()
        self.video_model = VideoCNN()
        self.caption_model = CaptionLSTM(vocab_size, embed_dim, hidden_dim, num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, video, caption):
        video_features = self.video_model(video)
        caption_features = self.caption_model(caption)
        combined = torch.cat((video_features, caption_features), dim=1)
        combined = self.dropout(combined)
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
test_data = create_test_data()
captions = [item["caption"] for item in all_data]
test_captions = [item["caption"] for item in test_data]
vocab = build_vocab(captions + test_captions)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])

dataset = BaseballDataset(all_data, vocab, transform=transform)
test_dataset_real = BaseballDataset(test_data, vocab, transform=transform)
test_dataset_notransform = BaseballDataset(test_data, vocab, transform=test_transforms)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, 
    collate_fn=lambda batch: (
        torch.stack([x[0] for x in batch]), 
        nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True), 
        torch.tensor([x[2] for x in batch])
    )
)

val_dataloader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, 
    collate_fn=lambda batch: (
        torch.stack([x[0] for x in batch]), 
        nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True), 
        torch.tensor([x[2] for x in batch])
    )
)

test_dataloader = DataLoader(
    test_dataset_real, batch_size=8, shuffle=False,
    collate_fn=lambda batch: (
        torch.stack([x[0] for x in batch]),
        nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True),
        torch.tensor([x[2] for x in batch])
    )
)

model = VideoCaptionClassifier(vocab_size=len(vocab), num_classes=len(all_labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

def evaluate(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_labels_list = []
    with torch.no_grad():
        for video, caption, label in dataloader:
            output = model(video, caption)
            loss = criterion(output, label)
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == label).sum().item()
            total += label.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_labels_list.extend(label.cpu().numpy())
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_labels_list, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels_list, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, f1

def visualize_prediction(model, test_dataset, test_dataset_no_transform, idx):
    """
    Visualize and predict the outcome for a single baseball pitch at a given index.
    
    Args:
        model: Trained model to use for prediction
        test_dataset: Test dataset containing the samples
        idx: Index of the sample in the test dataset to visualize
    """
    model.eval()
    
    # Get the sample at the specified index
    sample = test_dataset[idx]
    video, caption, label = sample
    
    # Add batch dimension for model input
    video_batch = video.unsqueeze(0)
    caption_batch = caption.unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        output = model(video_batch, caption_batch)
        probabilities = F.softmax(output, dim=1)
        pred_idx = torch.argmax(output, dim=1).item()

    sample_no_transform = test_dataset_no_transform[idx]
    video_no_transform, caption_no_transform, label_no_transform = sample_no_transform
    
    # Convert video tensor to displayable images
    frames = []
    for i in range(4):
        # Convert tensor to PIL image for display
        frame = video_no_transform[:, i, :, :]
        frame = frame.permute(1, 2, 0)  # Change from CxHxW to HxWxC
        frames.append(frame.numpy())
    
    # Get text labels
    pred_label = all_labels[pred_idx]
    true_label = all_labels[label.item()]
    
    # Extract tokens from caption tensor
    inv_vocab = {idx: word for word, idx in vocab.items()}
    caption_words = [inv_vocab.get(idx.item(), "<unk>") for idx in caption]
    caption_text = " ".join(caption_words)
    
    # Display results
    print(f"Caption: {caption_text}")
    print(f"True outcome: {true_label}")
    print(f"Predicted outcome: {pred_label}")
    print(f"Prediction confidence: {probabilities[0][pred_idx]:.4f}")
    
    # Display the video frames
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i, ax in enumerate(axes):
        ax.imshow(frames[i])
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
    
    plt.suptitle(f"Prediction: {pred_label} (True: {true_label})")
    plt.tight_layout()
    plt.show()


def train(model, train_dataset, test_dataloader, criterion, optimizer, epochs=15, batch_size=8):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    test_precisions, test_f1_scores = [], []
    
    weight_norms_history = []
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([x[0] for x in batch]), 
                nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True), 
                torch.tensor([x[2] for x in batch])
            )
        )
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=True)
        model.train()
        
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
        
        train_loss = total_loss / len(train_dataloader)
        train_accuracy = correct / total
        
        test_loss, test_accuracy, test_precision, test_f1 = evaluate(model, test_dataloader, criterion)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_f1_scores.append(test_f1)
        
        frame_weight_norms = model.video_model.get_weight_norms()
        weight_norms_history.append(frame_weight_norms)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}, "
              f"Test Precision: {test_precision:.2f}, Test F1: {test_f1:.2f}")
    
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
    plt.plot(epochs_range, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, test_precisions, label="Test Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.title("Precision vs Epoch")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, test_f1_scores, label="Test F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Epoch")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 5))
    
    weight_norms_history = torch.tensor(weight_norms_history) 
    for frame_idx in range(4):
        plt.plot(epochs_range, weight_norms_history[:, frame_idx], label=f"Conv2D-F{frame_idx}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Weight Norm (L2)")
    plt.title("Conv2D Weight Norms per Frame")
    plt.legend()
    plt.show()

def train_or_load_model(model, train_dataset, test_dataloader, criterion, optimizer, epochs=15, batch_size=8, model_path="model_weights.pth"):
    """Train the model or load from cache if available"""
    # Check if model weights exist
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path))
        # Evaluate the loaded model
        test_loss, test_accuracy, test_precision, test_f1 = evaluate(model, test_dataloader, criterion)
        print(f"Loaded model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}, "
                f"Test Precision: {test_precision:.2f}, Test F1: {test_f1:.2f}")
        return model
    
    # If no cached weights, train the model
    print(f"No cached weights found at {model_path}. Training model...")
    train(model, train_dataset, test_dataloader, criterion, optimizer, epochs, batch_size)
    
    # Save the model weights
    print(f"Saving model weights to {model_path}")
    torch.save(model.state_dict(), model_path)
    return model

train_or_load_model(model, train_dataset, test_dataloader, criterion, optimizer)
visualize_prediction(model, test_dataset_real, test_dataset_notransform, 5)