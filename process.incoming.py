import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
from openai import OpenAI
# from config import api_key


# Client = OpenAI(api_key=api_key) # Use when you'll have your own api key. 

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

def inference(prompt):
    print("Thinking...")
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response

# def inference_openai(prompt):
#     print("Thinking...")
#     response = clinet.responses.create(
#         model = "gpt-5",
#         input=prompt
#     )

#     return response.output_text   # use when you'll have your own api key and want to use OpenAI API instead of local LLM server. Make sure to set up your OPENAI_API_KEY in environment variables and install openai library. Also, change the model name in the inference function to an OpenAI model like "gpt-3.5-turbo" or "gpt-4".



df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# Find similarities of question_embedding with other embeddings.
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding'].shape))
similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]  # Reverse the order to get highest similarities first
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[["title", "number", "text"]])

prompt = f'''Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that :   

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------------
{incoming_query}
User asked this question related to the video chunks, you have to answer in a human way (don't mention the above format, it's just for you) where and how much content is taught in which (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrealted question, tell him that you can only answer questions related to the course.
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

# response = inference_openai(prompt)
# print(response)  # Use this if you want to use OpenAI API instead of local LLM server. Make sure to set up your OPENAI_API_KEY in environment variables and install openai library. Also, change the model name in the inference function to an OpenAI model like "gpt-3.5-turbo" or "gpt-4".

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item['title'], item['number'], item['text'], item['start'], item['end'])
