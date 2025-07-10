# Dynamic Hash Table

This project implements a scalable hash table in Python using **sorted buckets**, **binary search**, and **dynamic resizing** based on load factor. It simulates a real-world scenario by processing a dataset (`user_activity_data.csv`) containing user upload activity.

## 💡 Features

- Custom hash function
- Binary search within sorted buckets
- Dynamic resizing based on load factor (`alpha`)
- Load factor and resizing threshold are customizable
- Efficient search and insert operations
- CLI-based preview of bucket contents
- Shuffles and previews a sample from a real dataset

## 🗃 Sample Dataset

```plaintext
user123, 45
user456, 10
...
```

## 🧾 Files

- `DynamicHashTable` class
- CSV file parser with error handling
- Example to preview first 10 entries

## 📌 Usage

```bash
python dynamic_hash_table.py
```

## 🧠 Author

Syed Hasan Shahid  
GitHub: [@hasanshahid345](https://github.com/hasanshahid345)
