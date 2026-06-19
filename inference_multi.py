import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import re

print("Loading Multilingual Model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================
# 1. LOAD FINE-TUNED BERT MODELS
# ============================================
print("Loading Fine-Tuned IndoBERT...")
tokenizer_indo = AutoTokenizer.from_pretrained("model2/indobert_finetuned")
model_indo = AutoModel.from_pretrained("model2/indobert_finetuned")
model_indo.eval()
model_indo = model_indo.to(device)

print("Loading Fine-Tuned English BERT...")
tokenizer_en = AutoTokenizer.from_pretrained("model2/bert_en_finetuned")
model_en = AutoModel.from_pretrained("model2/bert_en_finetuned")
model_en.eval()
model_en = model_en.to(device)

# ============================================
# 2. LOAD CLASSIFIER ARTIFACTS
# ============================================
print("Loading classifier artifacts...")
tfidf = pickle.load(open("model2/tfidf_multi.pkl", "rb"))
label_encoder = pickle.load(open("model2/label_encoder_multi.pkl", "rb"))
scaler_tfidf = pickle.load(open("model2/scaler_tfidf_multi.pkl", "rb"))
scaler_bert = pickle.load(open("model2/scaler_bert_multi.pkl", "rb"))
df = pickle.load(open("model2/recipes_multi.pkl", "rb"))

# ============================================
# 3. LOAD NEURAL NETWORK MODEL
# ============================================
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

    def forward(self, x):
        x = self.fc1(x); x = torch.relu(x); x = self.bn1(x); x = self.dropout1(x)
        x = self.fc2(x); x = torch.relu(x); x = self.bn2(x); x = self.dropout2(x)
        x = self.fc3(x); x = torch.relu(x); x = self.bn3(x); x = self.dropout3(x)
        return self.fc4(x)

checkpoint = torch.load("model2/multilingual_model.pt", map_location=device)
model = MultilingualClassifier(checkpoint['input_dim'], checkpoint['num_classes']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Loaded! Classes: {checkpoint['num_classes']}")

# ============================================
# 4. HELPER FUNCTIONS
# ============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9/\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_language(text):
    """Detect language: id or en"""
    indo_words = ["ayam", "bawang", "garam", "nasi", "minyak", "air", "gula", "telur",
                  "sapi", "ikan", "udang", "tempe", "tahu", "cabai", "kunyit", "rebus",
                  "goreng", "bakar", "kukus", "tumis", "santan", "lengkuas", "jahe",
                  "daging", "bumbu", "kecap", "sambal", "rendang", "soto", "gado",
                  "merica", "ketumbar", "serai", "daun", "jeruk", "kemangi", "kemiri",
                  "penyedap", "presto", "ukep", "haluskan", "tiriskan", "diamkan",
                  "masukan", "tuangkan", "aduk", "koreksi", "sajikan", "hangat",
                  "potong", "cuci", "bersih", "peras", "geprek", "simpul"]
    count = sum(1 for word in indo_words if word in text.lower())
    return "id" if count >= 2 else "en"

def get_bert_embedding(text, lang):
    """Get BERT embedding from fine-tuned model based on detected language"""
    if lang == "id":
        tokenizer, bert_model = tokenizer_indo, model_indo
    else:
        tokenizer, bert_model = tokenizer_en, model_en

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        emb = outputs.pooler_output.cpu().numpy()

    emb_scaled = scaler_bert.transform(emb)
    return emb_scaled

def predict_category(text):
    """Predict category with auto language detection"""
    text_clean = clean_text(text)
    lang = detect_language(text_clean)

    # TF-IDF features
    tfidf_vec = tfidf.transform([text_clean]).toarray().astype(np.float32)
    tfidf_scaled = scaler_tfidf.transform(tfidf_vec)

    # BERT embedding (language-specific fine-tuned model)
    bert_emb = get_bert_embedding(text_clean, lang)

    # Combine and predict
    combined = np.hstack([tfidf_scaled, bert_emb]).astype(np.float32)
    tensor = torch.tensor(combined, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        conf = probs[0][idx].item()

    return label_encoder.inverse_transform([idx])[0], conf, lang

def predict_top_k(text, k=3):
    """Predict top-k categories"""
    text_clean = clean_text(text)
    lang = detect_language(text_clean)

    tfidf_vec = tfidf.transform([text_clean]).toarray().astype(np.float32)
    tfidf_scaled = scaler_tfidf.transform(tfidf_vec)
    bert_emb = get_bert_embedding(text_clean, lang)

    combined = np.hstack([tfidf_scaled, bert_emb]).astype(np.float32)
    tensor = torch.tensor(combined, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    top_k_probs, top_k_indices = torch.topk(probs, k)
    return [(label_encoder.inverse_transform([idx.item()])[0], prob.item()) 
            for prob, idx in zip(top_k_probs, top_k_indices)]

def recommend_recipe(query, top_n=5):
    """Content-based recommendation"""
    query_vec = tfidf.transform([clean_text(query)])
    recipe_matrix = tfidf.transform(df["text"])
    sim = cosine_similarity(query_vec, recipe_matrix)[0]
    top = sim.argsort()[-top_n:][::-1]
    return df.iloc[top][["title", "ingredients", "steps", "category"]]

# ============================================
# 5. INTERACTIVE MODE
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RECIPE CLASSIFIER - MULTILINGUAL (Fine-Tuned BERT)")
    print("="*60)
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("  Auto-detect language: Indonesia / English")
    print("  Commands: 'exit' = quit, '--topk' = show top-3")
    print("-"*60)

    while True:
        query = input("\nSearch: ").strip()
        if query.lower() == "exit":
            print("\nSampai jumpa!")
            break
        if not query:
            continue

        show_topk = "--topk" in query.lower()
        query = query.replace("--topk", "").strip()

        # Predict
        cat, conf, lang = predict_category(query)
        # lang_flag = "🇮🇩 ID" if lang == "id" else "🇬🇧 EN"

        # print(f"\n{lang_flag} | Category: {cat} (confidence: {conf:.2%})")
        print(f"\n(confidence: {conf:.2%})")

        if show_topk:
            print(f"\nTop 3 Predictions:")
            for i, (c, p) in enumerate(predict_top_k(query, k=3), 1):
                bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
                print(f"   {i}. {c:<15} {bar} {p:.2%}")

        # Recommendations
        print(f"\n Recommendations:")
        try:
            recs = recommend_recipe(query, top_n=5)
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                t = str(row['title']).title() if pd.notna(row['title']) else "?"
                c = str(row['category']) if pd.notna(row['category']) else "?"
                ing = str(row['ingredients'])[:120] if pd.notna(row['ingredients']) else ""
                print(f"\n   {i}. {t}")
                print(f"      {c} | {ing}...")
        except Exception as e:
            print(f"   Error: {e}")
        print("-"*60)