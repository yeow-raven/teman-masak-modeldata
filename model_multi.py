import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import os
import re
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.makedirs("model2", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================
# 1. LOAD DATASET
# ============================================
print("\nLoading datasets...")
df_indo = pd.read_csv("capstone/dataset_indo.csv")
df_en = pd.read_csv("capstone/dataset_en.csv")

print(f"Indonesian: {len(df_indo)} rows, {df_indo['category'].nunique()} categories")
print(f"English: {len(df_en)} rows, {df_en['category'].nunique()} categories")

# Combine for unified label encoding
df_all = pd.concat([df_indo, df_en], ignore_index=True)
print(f"Total: {len(df_all)} rows, {df_all['category'].nunique()} categories")

# ============================================
# 2. TF-IDF (for additional features)
# ============================================
print("\nBuilding TF-IDF...")
tfidf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1,2),
                        min_df=2, max_df=0.95, sublinear_tf=True, norm='l2')
X_tfidf = tfidf.fit_transform(df_all["text"]).toarray().astype(np.float32)

# ============================================
# 3. BERT FINE-TUNING SETUP
# ============================================
print("\nLoading BERT models for fine-tuning...")

MAX_LENGTH = 128
BATCH_SIZE = 8
BERT_LR = 2e-5
CLASSIFIER_LR = 1e-3
EPOCHS = 5
WARMUP_RATIO = 0.1

# IndoBERT
print("Loading IndoBERT...")
tokenizer_indo = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model_indo = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
model_indo = model_indo.to(device)

# English BERT
print("Loading English BERT...")
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
model_en = AutoModel.from_pretrained("bert-base-uncased")
model_en = model_en.to(device)

# Freeze early layers, unfreeze last 2 layers for fine-tuning
def freeze_bert_layers(model, num_unfreeze=2):
    """Freeze all layers except last N transformer layers"""
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        total_layers = len(model.encoder.layer)
        for i in range(total_layers - num_unfreeze, total_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True
            print(f"  Unfrozen layer {i}")

    # Unfreeze pooler
    if hasattr(model, 'pooler') and model.pooler is not None:
        for param in model.pooler.parameters():
            param.requires_grad = True
        print("  Unfrozen pooler")

print("\nFreezing IndoBERT layers (unfreeze last 2)...")
freeze_bert_layers(model_indo, num_unfreeze=2)

print("\nFreezing English BERT layers (unfreeze last 2)...")
freeze_bert_layers(model_en, num_unfreeze=2)

# Enable gradient checkpointing to save VRAM
model_indo.gradient_checkpointing_enable()
model_en.gradient_checkpointing_enable()

print("\nTrainable parameters:")
indo_trainable = sum(p.numel() for p in model_indo.parameters() if p.requires_grad)
en_trainable = sum(p.numel() for p in model_en.parameters() if p.requires_grad)
print(f"  IndoBERT: {indo_trainable:,}")
print(f"  BERT-EN: {en_trainable:,}")

# ============================================
# 4. LABEL ENCODING
# ============================================
label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(df_all["category"])
num_classes = len(label_encoder.classes_)
print(f"\nClasses: {num_classes} | {list(label_encoder.classes_)}")

# ============================================
# 5. CUSTOM DATASET FOR BERT
# ============================================
class RecipeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================
# 6. SPLIT DATA
# ============================================
# Get indices for each language
indo_indices = df_all[df_all["language"] == "id"].index.tolist()
en_indices = df_all[df_all["language"] == "en"].index.tolist()

# Split each language separately for stratification
indo_train_idx, indo_test_idx = train_test_split(
    indo_indices, test_size=0.2, random_state=42, 
    stratify=df_all.iloc[indo_indices]["category"]
)
en_train_idx, en_test_idx = train_test_split(
    en_indices, test_size=0.2, random_state=42,
    stratify=df_all.iloc[en_indices]["category"]
)

train_idx = indo_train_idx + en_train_idx
test_idx = indo_test_idx + en_test_idx

print(f"\nTrain: {len(train_idx)} | Test: {len(test_idx)}")

# ============================================
# 7. EXTRACT BERT EMBEDDINGS (with fine-tuning)
# ============================================
def extract_bert_embeddings_with_finetune(df_subset, indices, tokenizer, bert_model, 
                                         labels, epochs=5, batch_size=8, 
                                         model_name="BERT"):
    """Fine-tune BERT and extract embeddings"""

    texts = df_subset.iloc[indices]["text"].tolist()
    subset_labels = labels[indices]

    # Create dataset
    dataset = RecipeDataset(texts, subset_labels, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Classifier head for fine-tuning BERT
    class BertFineTuneHead(nn.Module):
        def __init__(self, hidden_size, num_classes):
            super().__init__()
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, pooled_output):
            x = self.dropout(pooled_output)
            return self.classifier(x)

    # Setup
    hidden_size = bert_model.config.hidden_size
    classifier_head = BertFineTuneHead(hidden_size, num_classes).to(device)

    # Optimizer - different LR for BERT and classifier
    bert_params = [p for p in bert_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': BERT_LR},
        {'params': classifier_head.parameters(), 'lr': CLASSIFIER_LR}
    ], weight_decay=0.01)

    # Scheduler
    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    # Mixed precision for VRAM efficiency
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Training loop
    bert_model.train()
    classifier_head.train()

    print(f"\nFine-tuning {model_name} for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                    pooled = outputs.pooler_output
                    logits = classifier_head(pooled)
                    loss = criterion(logits, labels_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.pooler_output
                logits = classifier_head(pooled)
                loss = criterion(logits, labels_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            total += labels_batch.size(0)
            correct += (pred == labels_batch).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100*correct/total:.1f}%'})

        avg_loss = total_loss / len(dataloader)
        acc = 100 * correct / total
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")

    # Extract embeddings (inference mode)
    bert_model.eval()
    embeddings = []

    inference_dataloader = DataLoader(
        RecipeDataset(texts, subset_labels, tokenizer, max_length=MAX_LENGTH),
        batch_size=batch_size * 2, shuffle=False
    )

    with torch.no_grad():
        for batch in tqdm(inference_dataloader, desc=f"Extracting {model_name} embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.pooler_output.cpu().numpy()
            embeddings.append(pooled)

    embeddings = np.vstack(embeddings)

    # Save fine-tuned BERT
    bert_model.save_pretrained(f"model2/{model_name.lower().replace(' ', '_')}_finetuned")
    tokenizer.save_pretrained(f"model2/{model_name.lower().replace(' ', '_')}_finetuned")

    return embeddings

# Fine-tune IndoBERT on Indonesian data
print("\n" + "="*60)
print("FINE-TUNING INDONESIAN BERT")
print("="*60)
emb_indo_train = extract_bert_embeddings_with_finetune(
    df_all, indo_train_idx, tokenizer_indo, model_indo, 
    y_all, epochs=EPOCHS, batch_size=BATCH_SIZE, model_name="IndoBERT"
)

emb_indo_test = extract_bert_embeddings_with_finetune(
    df_all, indo_test_idx, tokenizer_indo, model_indo,
    y_all, epochs=1, batch_size=BATCH_SIZE * 2, model_name="IndoBERT_test"
)

# Fine-tune English BERT on English data
print("\n" + "="*60)
print("FINE-TUNING ENGLISH BERT")
print("="*60)
emb_en_train = extract_bert_embeddings_with_finetune(
    df_all, en_train_idx, tokenizer_en, model_en,
    y_all, epochs=EPOCHS, batch_size=BATCH_SIZE, model_name="BERT_EN"
)

emb_en_test = extract_bert_embeddings_with_finetune(
    df_all, en_test_idx, tokenizer_en, model_en,
    y_all, epochs=1, batch_size=BATCH_SIZE * 2, model_name="BERT_EN_test"
)

# Combine embeddings
print("\nCombining embeddings...")
emb_dim = model_indo.config.hidden_size

X_bert_train = np.zeros((len(train_idx), emb_dim), dtype=np.float32)
X_bert_test = np.zeros((len(test_idx), emb_dim), dtype=np.float32)

# Fill train embeddings
for i, idx in enumerate(indo_train_idx):
    train_pos = train_idx.index(idx)
    X_bert_train[train_pos] = emb_indo_train[i]

for i, idx in enumerate(en_train_idx):
    train_pos = train_idx.index(idx)
    X_bert_train[train_pos] = emb_en_train[i]

# Fill test embeddings
for i, idx in enumerate(indo_test_idx):
    test_pos = test_idx.index(idx)
    X_bert_test[test_pos] = emb_indo_test[i]

for i, idx in enumerate(en_test_idx):
    test_pos = test_idx.index(idx)
    X_bert_test[test_pos] = emb_en_test[i]

# ============================================
# 8. SCALE AND COMBINE FEATURES
# ============================================
print("\nScaling features...")

scaler_tfidf = StandardScaler()
scaler_bert = StandardScaler()

X_tfidf_train = X_tfidf[train_idx]
X_tfidf_test = X_tfidf[test_idx]

X_tfidf_train_scaled = scaler_tfidf.fit_transform(X_tfidf_train)
X_tfidf_test_scaled = scaler_tfidf.transform(X_tfidf_test)

X_bert_train_scaled = scaler_bert.fit_transform(X_bert_train)
X_bert_test_scaled = scaler_bert.transform(X_bert_test)

X_train = np.hstack([X_tfidf_train_scaled, X_bert_train_scaled]).astype(np.float32)
X_test = np.hstack([X_tfidf_test_scaled, X_bert_test_scaled]).astype(np.float32)

print(f"Combined train shape: {X_train.shape}")
print(f"Combined test shape: {X_test.shape}")

# ============================================
# 9. PREPARE TENSORS
# ============================================
y_train = y_all[train_idx]
y_test = y_all[test_idx]

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ============================================
# 10. CLASSIFIER MODEL
# ============================================
print("\nTraining final classifier...")

class MultilingualClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultilingualClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x); x = torch.relu(x); x = self.bn1(x); x = self.dropout1(x)
        x = self.fc2(x); x = torch.relu(x); x = self.bn2(x); x = self.dropout2(x)
        x = self.fc3(x); x = torch.relu(x); x = self.bn3(x); x = self.dropout3(x)
        return self.fc4(x)

model = MultilingualClassifier(X_train.shape[1], num_classes).to(device)
print(f"Classifier parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 11. TRAINING CLASSIFIER
# ============================================
criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

best_val_acc = 0
patience = 15
patience_counter = 0
best_model_state = None
min_epochs = 80

for epoch in range(100):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (pred == batch_y).sum().item()

    train_acc = 100 * train_correct / train_total

    model.eval()
    with torch.no_grad():
        val_out = model(X_test_tensor)
        val_loss = criterion(val_out, y_test_tensor).item()
        _, val_pred = torch.max(val_out, 1)
        val_acc = 100 * (val_pred == y_test_tensor).sum().item() / len(y_test_tensor)

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}] Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%")

    if epoch + 1 >= min_epochs and patience_counter >= patience:
        print(f"Early stop epoch {epoch+1}. Best: {best_val_acc:.2f}%")
        break

if best_model_state:
    model.load_state_dict(best_model_state)

# ============================================
# 12. EVALUATION
# ============================================
model.eval()
with torch.no_grad():
    test_out = model(X_test_tensor)
    _, test_pred = torch.max(test_out, 1)
    test_pred_cpu = test_pred.cpu().numpy()
    y_test_cpu = y_test_tensor.cpu().numpy()

acc = accuracy_score(y_test_cpu, test_pred_cpu)
print(f"\n{'='*50}")
print(f"TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
print(f"{'='*50}")

print("\nClassification Report:")
print(classification_report(y_test_cpu, test_pred_cpu, 
                            labels=np.arange(num_classes),
                            target_names=label_encoder.classes_,
                            digits=3, zero_division=0))

# ============================================
# 13. SAVE
# ============================================
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': X_train.shape[1],
    'num_classes': num_classes,
    'classes': label_encoder.classes_.tolist()
}, "model2/multilingual_model.pt")

pickle.dump(tfidf, open("model2/tfidf_multi.pkl", "wb"))
pickle.dump(label_encoder, open("model2/label_encoder_multi.pkl", "wb"))
pickle.dump(scaler_tfidf, open("model2/scaler_tfidf_multi.pkl", "wb"))
pickle.dump(scaler_bert, open("model2/scaler_bert_multi.pkl", "wb"))
pickle.dump(df_all, open("model2/recipes_multi.pkl", "wb"))

with open("model2/classes_multi.txt", "w") as f:
    for cls in label_encoder.classes_:
        f.write(f"{cls}\n")

print("\n✅ Saved to model2/:")
print("   - multilingual_model.pt")
print("   - tfidf_multi.pkl")
print("   - label_encoder_multi.pkl")
print("   - scaler_tfidf_multi.pkl")
print("   - scaler_bert_multi.pkl")
print("   - recipes_multi.pkl")
print("   - indobert_finetuned/")
print("   - bert_en_finetuned/")