#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Tokenizer:    
    def __init__(self):
        self.vocab = {}
        self.rev_vocab = {}    
    
    def fit(self, texts):
        index = 0   

        for text in texts:      
            text = text.lower()
            for punct in ".,!?;:-()[]{}'\"":
                text = text.replace(punct, "")  
            tokens = text.split()        
            
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = index
                    self.rev_vocab[index] = token
                    index += 1
                    
    def transform(self, texts):
        if not self.vocab:
            raise ValueError("Tokenizer has not been fit yet. Please call fit() before transform().")
        
        vectors = []
        for text in texts:
            text = text.lower()
            for punct in ".,!?;:-()[]{}'\"":
                text = text.replace(punct, "")
            tokens = text.split()
            vector = [0] * len(self.vocab)
            for token in tokens:
                if token in self.vocab:
                    idx = self.vocab[token]
                    vector[idx] = 1
            vectors.append(vector)
        return vectors

    def inverse_transform(self, vectors):
        if not self.rev_vocab:
            raise ValueError("Tokenizer has not been fit yet. Please call fit() before inverse_transform().")

        results = []

        for vector in vectors:
            tokens = []
            for index, value in enumerate(vector):
                if value == 1 and index in self.rev_vocab:
                    tokens.append(self.rev_vocab[index])
            results.append(tokens)

        return results
    
tokenizer = Tokenizer()

training_texts = ["I love apples!", "You eat bananas", "I eat apples and bananas!"]
tokenizer.fit(training_texts)

print("Vocab:", tokenizer.vocab)
print("Reverse Vocab:", tokenizer.rev_vocab)

test_texts = ["I eat bananas", "You love apples"]
vectors = tokenizer.transform(test_texts)
print("Multi-hot Vectors:", vectors)

tokens = tokenizer.inverse_transform(vectors)
print("Recovered Tokens:", tokens)

