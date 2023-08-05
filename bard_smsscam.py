import os
import pandas as pd
from tqdm import tqdm
import time


from bardapi import Bard

token = 'WwgVaiTy3rvuiJXunITntxgBdvu8YfQvhLIh3DvoGa7ZCQdGVXntSU8_pDx3EJ4B3GVXOg.'
bard = Bard(token=token)
bard.get_answer("나와 내 동년배들이 좋아하는 뉴진스에 대해서 알려줘")['content']

def scam_bard_checker(input_text):
    
    # prompt = f"""
    # I want you to act like sms scammer detector. I will give you a sentence. You must answer whether that sentence is scam or not. You must only answer the label: "1" for scam or "0". No more. the text is:
    #     "{input_text}""."
    # """
    
    prompt=f"""
           Read the text. Only answer 1 for scam, 0 otherwise. No explaination. text:"{input_text}"."
           """
    
    return bard.get_answer(prompt)['content']

retry_delay = 30
max_retry = 5
df = pd.read_csv("data/sms_spam.csv",encoding = "ISO-8859-1")
res = pd.DataFrame(columns=['text', 'truth', 'predict', 'tokens'])
for index, row in tqdm(df.iterrows(), total=len(df)):
    i=0
    label = row['v1']
    text = row['v2']
    while True and (i<max_retry):
        try:
            predict = scam_bard_checker(text)
        except Exception as e:
            i+=1
            print("Retrying in", retry_delay, "seconds...")
            time.sleep(retry_delay)
            continue
        break
        
    # res = res.append({'text':text, 'truth':label, 'predict':predict, 'tokens':tokens}, ignore_index=True)
    with open("data/bard_sms_spam_res.txt", "a+") as f:
        f.write(f"{text},{label},{predict}\n")
                
    res = pd.concat([res, pd.DataFrame([[text, label, predict]], columns=['text', 'truth', 'predict'])],axis=0)
    
res.to_csv("data/bard_sms_spam_res.csv", index=True)