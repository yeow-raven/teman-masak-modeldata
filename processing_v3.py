import pandas as pd
import emoji
import re
import os

def clean_ingredients(text):
    """Clean ingredient text - preserve measurements, fix separators, handle abbreviations"""
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace="")

    # Pisahin angka yang nempel langsung sama satuan (50gr -> 50 gr, 1sdt -> 1 sdt)
    # biar abbreviation expansion di bawah bisa kebaca
    text = re.sub(r'(\d)(gr|kg|ml|ltr|sdm|sdt|bh|pcs|btg|lbr|ds|sdk)\b', r'\1 \2', text)

    # Fix abbreviations FIRST (before removing special chars)
    abbreviations = {
        r'\bbwg\b': 'bawang',
        r'\bbaput\b': 'bawang putih',
        r'\bbamer\b': 'bawang merah',
        r'\bbawang putih\b': 'bawang putih',  # keep as is
        r'\bbawang merah\b': 'bawang merah',  # keep as is
        r'\bcabe\b': 'cabai',
        r'\bsdm\b': 'sendok makan',
        r'\bsdt\b': 'sendok teh',
        r'\bgr\b': 'gram',
        r'\bkg\b': 'kilogram',
        r'\bml\b': 'mililiter',
        r'\bltr\b': 'liter',
        r'\bl\b(?=\s)': 'liter',
        r'\bpcs\b': 'potong',
        r'\bbh\b': 'buah',
        r'\bbtg\b': 'batang',
        r'\blbr\b': 'lembar',
        r'\bds\b': 'sendok',
        r'\bsdk\b': 'sendok',
    }

    for abbr, full in abbreviations.items():
        text = re.sub(abbr, full, text)

    # Replace "dan" with comma for ingredient separation (but not in numbers like "2 dan 3")
    text = re.sub(r'(?<=[a-z])\s+dan\s+(?=[a-z])', ', ', text)

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Replace multiple dashes (--) with comma + space (main ingredient separator in Indonesian recipes)
    text = re.sub(r"-{2,}", ", ", text)

    # Dash antar angka (50-100, 2-3) itu range takaran, JANGAN dihilangkan,
    # biarin aja biar gak ambigu

    # Keep: alphanumeric, comma, slash (for fractions), whitespace
    # Remove: parentheses, brackets, other special chars, but KEEP comma and slash
    # text = re.sub(r"[^a-zA-Z0-9/,\s]", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove leading/trailing commas
    text = re.sub(r"^,\s*", "", text)
    text = re.sub(r"\s*,\s*$", "", text)

    # Fix multiple commas
    text = re.sub(r",\s*,", ",", text)

    return text

def clean_steps(text):
    """Clean steps text - preserve numbering, remove special chars, use semicolon as separator"""
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace="")

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Replace periods with semicolons to separate steps (avoid mixing with ingredient commas)
    text = text.replace('.', '; ')

    # Keep: alphanumeric, whitespace, semicolon, numbers
    # Remove: parentheses, brackets, other special chars except semicolon
    text = re.sub(r"[^a-zA-Z0-9;\s]", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Fix multiple semicolons
    text = re.sub(r";\s*;", ";", text)

    return text

def clean_title(text):
    """Clean title text - simple cleaning"""
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_language(text):
    """Detect if text is Indonesian or English"""
    indo_words = ["ayam", "bawang", "garam", "nasi", "minyak", "air", "gula", "telur",
                  "sapi", "ikan", "udang", "tempe", "tahu", "cabai", "kunyit", "rebus",
                  "goreng", "bakar", "kukus", "tumis", "santan", "lengkuas", "jahe",
                  "daging", "bumbu", "kecap", "sambal", "rendang", "soto", "gado",
                  "merica", "ketumbar", "serai", "daun", "jeruk", "kemangi", "kemiri",
                  "penyedap", "presto", "ukep", "haluskan", "tiriskan", "diamkan",
                  "masukan", "tuangkan", "aduk", "koreksi", "sajikan", "hangat",
                  "potong", "cuci", "bersih", "peras", "geprek", "simpul",
                  "sendok", "gram", "kilogram", "liter", "mililiter", "potong", "buah",
                  "batang", "lembar", "secukupnya", "geprek", "iris", "cincang"]
    count = sum(1 for word in indo_words if word in text.lower())
    return "id" if count >= 2 else "en"

# Load datasets
print("Loading datasets...")
df1 = pd.read_csv("capstone/recipes_extended.csv")
df2 = pd.read_csv("capstone/Indonesian_Food_Recipes.csv")

print(f"English dataset columns: {list(df1.columns)}")
print(f"Indonesian dataset columns: {list(df2.columns)}")

# ============================================
# SELECT AND STANDARDIZE COLUMNS
# ============================================
# English dataset: recipe_title, category, ingredient_text, directions_text
df1 = df1[["recipe_title", "ingredient_text", "directions_text", "category"]].copy()
df1.columns = ["title", "ingredients", "steps", "category"]

# Indonesian dataset: Title, Ingredients (RAW, masih lengkap sama takaran), Steps, Category
# JANGAN pake "Ingredients Cleaned" -> kolom itu di dataset aslinya udah dihapus takarannya
df2 = df2[["Title", "Ingredients", "Steps", "Category"]].copy()
df2.columns = ["title", "ingredients", "steps", "category"]

# Combine
df = pd.concat([df1, df2], ignore_index=True)
df.dropna(subset=["title", "ingredients", "steps", "category"], inplace=True)
print(f"\nCombined: {len(df)} rows")

# ============================================
# CLEAN TEXT WITH APPROPRIATE FUNCTION FOR EACH COLUMN
# ============================================
df["title"] = df["title"].apply(clean_title)
df["ingredients"] = df["ingredients"].apply(clean_ingredients)
df["steps"] = df["steps"].apply(clean_steps)

# ============================================
# CATEGORY MAPPING (comprehensive for both)
# ============================================
category_map = {
    # Indonesian categories
    "ayam": "Unggas", "tempe": "Nabati", "tahu": "Nabati",
    "udang": "Seafood", "ikan": "Seafood", "sapi": "Daging",
    "kambing": "Daging", "telur": "Telur",

    # English main categories -> mapped to Indonesian categories
    "Main Dishes": "Hidangan Utama", "Dinner": "Hidangan Utama",
    "Beef Recipes": "Daging", "Pork": "Daging", "Beef Stews": "Daging",
    "Chicken Salads": "Unggas", "Fried Chicken": "Unggas",
    "Crab Cakes": "Seafood", "Chowders": "Seafood",
    "Desserts": "Dessert", "Cakes": "Dessert", "Cookies": "Dessert",
    "Breads": "Roti", "Banana Breads": "Roti",
    "Breakfast And Brunch": "Sarapan", "Pancakes": "Sarapan",
    "Appetizers And Snacks": "Camilan", "Doughnuts": "Camilan",
    "Cocktails": "Minuman", "Lemonade": "Minuman",
    "Healthy Recipes": "Sehat", "Vegetarian": "Sehat",
    "Mexican": "Internasional", "Chinese": "Internasional", "Italian": "Internasional",
    "Air Fryer Recipes": "Hidangan Utama", "Salads": "Sehat",
    "Soups And Stews": "Hidangan Utama", "Pasta": "Hidangan Utama",
    "Side Dishes": "Hidangan Utama", "Drinks": "Minuman",
    "Lunch": "Hidangan Utama", "World Cuisine": "Internasional",
    "Fruit": "Dessert", "Pie": "Dessert", "Brownies": "Dessert",
    "Ice Cream": "Dessert", "Pudding": "Dessert",
    "Smoothies": "Minuman", "Coffee": "Minuman", "Tea": "Minuman",
    "Sandwiches": "Camilan", "Wraps": "Camilan", "Pizza": "Camilan",

    # Additional English categories that should be mapped
    "Allrecipes Allstar Recipes": "Hidangan Utama",
    "Apple Pie": "Dessert",
    "Baked Beans": "Nabati",
    "Bar Cookies": "Dessert",
    "Biscuits": "Roti",
    "Blueberry Pie": "Dessert",
    "Breakfast Casseroles": "Sarapan",
    "Breakfast Potatoes": "Sarapan",
    "Buffalo Chicken Wings": "Unggas",
    "Burgers": "Hidangan Utama",
    "Burritos": "Internasional",
    "Butternut Squash Soups": "Hidangan Utama",
    "Camping Recipes": "Hidangan Utama",
    "Canning And Preserving": "Camilan",
    "Casseroles": "Hidangan Utama",
    "Cheese Balls": "Camilan",
    "Cheesecakes": "Dessert",
    "Chef John": "Hidangan Utama",
    "Cherry Pie": "Dessert",
    "Chicken Noodle Soups": "Hidangan Utama",
    "Chicken Teriyaki": "Unggas",
    "Chili Recipes": "Hidangan Utama",
    "Chocolate Cakes": "Dessert",
    "Chocolate Chip Cookies": "Dessert",
    "Chocolate Fudge": "Dessert",
    "Christmas": "Dessert",
    "Christmas Cookies": "Dessert",
    "Cobblers": "Dessert",
    "Coleslaws": "Sehat",
    "Comfort Food": "Hidangan Utama",
    "Cooking For A Crowd": "Hidangan Utama",
    "Cooking For One": "Hidangan Utama",
    "Cooking For Two": "Hidangan Utama",
    "Cranberry Sauces": "Camilan",
    "Crisps And Crumbles": "Dessert",
    "Diabetic": "Sehat",
    "Diwali": "Internasional",
    "Dumplings": "Camilan",
    "Easter": "Dessert",
    "Enchiladas": "Internasional",
    "Father'S Day": "Hidangan Utama",
    "Fettuccini": "Hidangan Utama",
    "Food Gifts": "Camilan",
    "Fried Rice": "Hidangan Utama",
    "Fries": "Camilan",
    "Frittatas": "Sarapan",
    "Frostings And Icings": "Dessert",
    "Fruit Salads": "Dessert",
    "Gluten-Free": "Sehat",
    "Gravies": "Hidangan Utama",
    "Green Salads": "Sehat",
    "Ground Beef": "Daging",
    "Halloween": "Dessert",
    "Hanukkah": "Dessert",
    "High Fiber": "Sehat",
    "Indian": "Internasional",
    "Instant Pot": "Hidangan Utama",
    "Jewish": "Hidangan Utama",
    "July 4Th": "Hidangan Utama",
    "Kosher": "Hidangan Utama",
    "Labor Day": "Hidangan Utama",
    "Lamb": "Daging",
    "Lasagna": "Hidangan Utama",
    "Leftovers": "Hidangan Utama",
    "Linguine": "Hidangan Utama",
    "Low Calorie": "Sehat",
    "Low Fat": "Sehat",
    "Low Sodium": "Sehat",
    "Macaroni And Cheese": "Hidangan Utama",
    "Mardi Gras": "Hidangan Utama",
    "Muffins": "Roti",
    "Mushroom Soups": "Hidangan Utama",
    "Mushrooms": "Nabati",
    "New Year'S": "Hidangan Utama",
    "Paleo": "Sehat",
    "Pasta Salads": "Hidangan Utama",
    "Pie Crusts": "Dessert",
    "Pies": "Dessert",
    "Winter Squash": "Nabati",
}

df["category"] = df["category"].map(category_map).fillna(df["category"])

# ============================================
# DETECT LANGUAGE
# ============================================
df["language"] = df["title"].apply(detect_language)
print(f"\nLanguage distribution:")
print(df["language"].value_counts())

# ============================================
# SPLIT BY LANGUAGE
# ============================================
df_indo = df[df["language"] == "id"].copy()
df_en = df[df["language"] == "en"].copy()

# ============================================
# CREATE WEIGHTED TEXT
# ============================================
# Title weighted 3x, ingredients 2x, steps 1x
# Use | as separator between sections to avoid confusion
df_indo["text"] = (df_indo["title"] + " | ") * 3 + (df_indo["ingredients"] + " | ") * 2 + df_indo["steps"]
df_en["text"] = (df_en["title"] + " | ") * 3 + (df_en["ingredients"] + " | ") * 2 + df_en["steps"]

# Remove duplicates
df_indo.drop_duplicates(subset=["text"], inplace=True)
df_en.drop_duplicates(subset=["text"], inplace=True)
df_indo.reset_index(drop=True, inplace=True)
df_en.reset_index(drop=True, inplace=True)

# ============================================
# FILTER CATEGORIES WITH < 50 SAMPLES
# ============================================
print(f"\nBefore filtering:")
print(f"  Indonesian: {len(df_indo)} rows, {df_indo['category'].nunique()} categories")
print(f"  English: {len(df_en)} rows, {df_en['category'].nunique()} categories")

# Filter Indonesian
counts_id = df_indo["category"].value_counts()
valid_id = counts_id[counts_id >= 50].index
df_indo = df_indo[df_indo["category"].isin(valid_id)].copy()

# Filter English
counts_en = df_en["category"].value_counts()
valid_en = counts_en[counts_en >= 50].index
df_en = df_en[df_en["category"].isin(valid_en)].copy()

print(f"\nAfter filtering (min 50 samples per category):")
print(f"  Indonesian: {len(df_indo)} rows, {df_indo['category'].nunique()} categories")
print(f"  English: {len(df_en)} rows, {df_en['category'].nunique()} categories")

print(f"\nIndonesian categories: {sorted(df_indo['category'].unique())}")
print(f"English categories: {sorted(df_en['category'].unique())}")

# ============================================
# SAVE
# ============================================
os.makedirs("capstone", exist_ok=True)

df_indo.to_csv("capstone/dataset_indo.csv", index=False)
df_en.to_csv("capstone/dataset_en.csv", index=False)

df_merged = pd.concat([df_indo, df_en], ignore_index=True)
df_merged.to_csv("capstone/dataset_merged.csv", index=False)

print(f"\nSaved:")
print(f"   - capstone/dataset_indo.csv ({len(df_indo)} rows)")
print(f"   - capstone/dataset_en.csv ({len(df_en)} rows)")
print(f"   - capstone/dataset_merged.csv ({len(df_merged)} rows)")

# Show sample of cleaned data
print(f"\nSample Indonesian data:")
for idx in range(min(2, len(df_indo))):
    row = df_indo.iloc[idx]
    print(f"\nTitle: {row['title']}")
    print(f"Ingredients: {row['ingredients']}")
    print(f"Category: {row['category']}")

print(f"\nSample English data:")
for idx in range(min(2, len(df_en))):
    row = df_en.iloc[idx]
    print(f"\nTitle: {row['title']}")
    print(f"Ingredients: {row['ingredients']}")
    print(f"Category: {row['category']}")