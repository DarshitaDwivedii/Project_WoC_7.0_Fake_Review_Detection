# ✨ **Checkpoint 1: Data Preprocessing** ✨  

---

## 🛠️ **About this Checkpoint**  

This phase focuses on cleaning and preprocessing the raw dataset to prepare it for analysis and model training. The process involves removing noise, normalizing text, and transforming data into a usable format.  

---

## 📄 **Steps in Data Preprocessing**  

1. **Language Detection**: Non-English reviews are filtered out. 🌍  
2. **Contraction Expansion**: Words like *"don't"* are expanded to *"do not"*.  
3. **Emoji Normalization**: Emojis are converted to text using `emoji.demojize`. 😄 → `:smile:`  
4. **Text Cleaning**:  
   - HTML tags, URLs, and special characters are removed.  
   - Unnecessary spaces and numbers are stripped out.  
5. **Tokenization**: Sentences are split into individual words. 🔤  
6. **Stopword Removal**: Common words like *"the", "and"* are excluded.  
7. **Lemmatization**: Words are reduced to their base form (*e.g., "running" → "run"*).  

---

## 📂 **Folder Structure**  

```plaintext  
📂 checkpoint/  
├── preprocessing.py        # Script for data preprocessing  
├── preprocessed_data/      # Cleaned and processed dataset  
├── README.md               # Documentation for this checkpoint  
```  

---

## 💡 **Importance of Preprocessing**  

- **Improves Model Accuracy**: Ensures clean input for better predictions.  
- **Reduces Noise**: Removes irrelevant or redundant data.  
- **Feature Engineering**: Converts raw text into meaningful numerical data.  

---

> "Clean data leads to accurate insights."  

---  
