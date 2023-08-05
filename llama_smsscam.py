# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama

from tqdm import tqdm
import pandas as pd


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
    df = pd.read_csv("../data/sms_spam.csv",encoding = "ISO-8859-1")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        label = row['v1']
        text = row['v2']
        # print(text)
        dialog = [
            {
                "role": "system",
                "content": "Always anwser 1 or 0. No explain",
            },
            {"role": "user", "content": f"""
            Act as SMS scam detector. 1 for scam, 0 for ham. Text:"{text}"."
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
        with open("../data/llama_sms_spam_res.txt", "a+") as f:
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
