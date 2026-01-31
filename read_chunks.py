import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list, batch_size=16):
    all_embeddings = []


# Process the text_list in batches. This helps to avoid overwhelming the API with too many requests at once. And it can also help to stay within any rate limits that the API might have.
    for i in range(0, len(text_list), batch_size): 
        batch = text_list[i:i + batch_size]
        
        r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": batch
    })

        data = r.json()

    # print("API response:", data)   # Debugging line to see the full response that is causing the error/issue

# Check if the response contains embeddings. If not, skip this batch and log the error.
        if "embeddings" not in data:
            print(f"Skipping batch {i // batch_size} due to embedding error: {data}")
            continue
    
        all_embeddings.extend(data["embeddings"])
        
    return all_embeddings
    
    # embedding = r.json()["embeddings"] 
    # return embedding


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        if i >= len(embeddings):
            break
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
# print(df)

# Save this dataframe
joblib.dump(df, "embeddings.joblib")








# Alternative simpler version without batching (may run into issues with large inputs)
'''
import requests
import os
import json
import pandas as pd

def create_embedding(text_list):
     r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

jsons = os.listdir("jsons") # List all the jsons
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
print(df)

'''