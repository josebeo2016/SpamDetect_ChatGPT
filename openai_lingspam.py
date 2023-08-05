import openai
import os
import pandas as pd
from tqdm import tqdm
import time
import tiktoken

openai.api_key = os.getenv('OPENAI_API_KEY')

def limit_text(text: str, max_tokens: int) -> str:
    """Returns the text, limiting it to the specified number of tokens."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encode = encoding.encode(text)
    if len(encode) <= max_tokens:
        return text
    else:
        return encoding.decode(encode[:max_tokens])
def get_completion(prompt, model="gpt-3.5-turbo", temperature=0): 
    messages = [{"role": "user", "content": prompt}]
    # count = num_tokens_from_string(prompt)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
    )
    # print(response)
    return response.choices[0].message["content"], response.usage.total_tokens

def scammer_check(input_text):
    
    # prompt = f"""
    #     I want you to act like sms scammer detector. I will give you a sentence. You must answer whether that sentence is scam or not. You must only answer the label: "1" for scam or "0". No explain. the text is:
    #     "{input_text}""."
    # """
    prompt = f"""
    Act as SMS scam detector. 1 for scam, 0 otherwise. Text:"{input_text}"."
    """
    
    response, count = get_completion(prompt)
    return response, count

retry_delay = 30
max_retry = 5
df = pd.read_csv("data/lingspam.csv",encoding = "ISO-8859-1")
res = pd.DataFrame(columns=['ID', 'truth', 'predict', 'tokens'])
for index, row in tqdm(df[406:].iterrows(), total=len(df[406:])):
    i=0
    label = 'spam' if int(row['label'])==1 else 'ham'
    text = row['message']
    if type(text) is not str:
        print("Text is None")
        with open("data/openai_lingspam_res.txt", "a+") as f:
            f.write(f"{index},{label},-1,-1\n")
        continue
    text = limit_text(text, 1000)
        
    while True and (i<max_retry):
        try:
            predict, tokens = scammer_check(text)
        except Exception as e:
            print(e)
            i+=1
            print("Retrying in", retry_delay, "seconds...")
            time.sleep(retry_delay)
            continue
        break
        
    # res = res.append({'text':text, 'truth':label, 'predict':predict, 'tokens':tokens}, ignore_index=True)
    with open("data/openai_lingspam_res.txt", "a+") as f:
        f.write(f"{index},{label},{predict},{tokens}\n")
                
    res = pd.concat([res, pd.DataFrame([[index, label, predict, tokens]], columns=['ID', 'truth', 'predict', 'tokens'])],axis=0)
    
res.to_csv("data/openai_lingspam_res.csv", index=False)