# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama

from tqdm import tqdm
import pandas as pd
import tiktoken

def limit_text(text: str, max_tokens: int) -> str:
    """Returns the text, limiting it to the specified number of tokens."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encode = encoding.encode(text)
    if len(encode) <= max_tokens:
        return text
    else:
        return encoding.decode(encode[:max_tokens])
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    # dialogs = []
    # load scam data
    df = pd.read_csv("../data/enron_spam_data.csv",encoding = "ISO-8859-1", index_col=0)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        label = row['Spam/Ham']
        text = row['Message']
        if type(text) is not str:
            print("Text is None")
            with open("../data/llama_enron_res.txt", "a+") as f:
                f.write(f"{index},{label},-1,-1\n")
            continue
        text = limit_text(text, 250)
        
        # print(text)
        dialog = [
            {
                "role": "system",
                "content": "Always anwser 1 or 0. No explain",
            },
            {"role": "user", "content": f"""
            Act as SMS spam detector. 1 for spam, 0 for ham. Text:"{text}"."
            """},
        ]
        # print(dialog)

        results = generator.chat_completion(
            [dialog],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
            
        # print(dialog[1]['content'])
        # print(results[0]['generation']['content'])
        # print("\n==================================\n")
        predict = results[0]['generation']['content'].strip().replace("\n", "")
        with open("../data/llama_enron_res.txt", "a+") as f:
            f.write(f"{index},{label},{predict}\n")

    # for dialog, result in zip(dialogs, results):
    #     for msg in dialog:
    #         print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    #     print(
    #         f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    #     )
    #     print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
