
import json
from typing import List, Literal, TypedDict


Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
def format_tokens(dialog, tokenizer):
    prompt_tokens = []
    if dialog[0]["role"] == "system":   #如果是 system ，由於llama沒有這個，所以去掉
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS        #system信息的開始
                + dialog[0]["content"]  #system内容
                + E_SYS                 #system結尾
                + dialog[1]["content"],
            }
        ]

    dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                ) + [tokenizer.eos_token_id]
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
    assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
    prompt_tokens.append(dialog_tokens)
    return prompt_tokens

        

def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs
