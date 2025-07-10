# Tokenizer Development 🧠

This project implements a **custom tokenizer** in Python for converting raw text into **multi-hot encoded vectors** and reversing them back into tokens.

## 🔍 Features

- Case-insensitive vocabulary creation
- Punctuation removal
- Multi-hot vector transformation
- Reverse transformation (decode vectors back to words)
- Works on multiple texts

## 💡 How It Works

1. `fit(texts)` – Builds the vocabulary from training texts.
2. `transform(texts)` – Converts new texts into multi-hot vectors using the vocabulary.
3. `inverse_transform(vectors)` – Converts the vectors back to token lists.

## 🧪 Example

```python
texts = ["I love apples!", "You eat bananas"]
tokenizer.fit(texts)
vectors = tokenizer.transform(["I eat bananas"])
tokens = tokenizer.inverse_transform(vectors)
```

## 🔢 Sample Output

```plaintext
Vocab: {'i': 0, 'love': 1, 'apples': 2, ...}
Multi-hot Vectors: [[1, 0, 0, 1, 1], [0, 1, 1, 0, 0]]
Recovered Tokens: [['i', 'eat', 'bananas'], ['you', 'love', 'apples']]
```

## 🧠 Author

Syed Hasan Shahid  
GitHub: [@hasanshahid345](https://github.com/hasanshahid345)
