import os
import random
from chat_utils import format_tokens
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset

import json

import numpy as np
import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


from env import *
from worker import RolloutWorker
from agent import Agents
from replay_buffer import ReplayBuffer
from eval_gsm import Eval_Gsm
from loaddata import arg_parser
from utils_extra import set_random_seed,create_dataloader

from glob import glob
import pandas as pd



# 方法用于从文件中获得各个文本
def response_saved_data(generated_description):
    states, questions, responses, observations, standard_answers = [], [], [], [], []
    response_dict = generated_description
    questions = list(response_dict.keys())

    states, standard_answer = response_dict[question]
    standard_answers.append(standard_answer)
    for state in states:
        for i in range(len(state)):
            if i == 2:
                observations.append(state[i]['content'])
            if i == 3:
                start_index = state[i]['content'].find(". [/INST]")
                pred_solution = state[i]['content'][start_index + 11:]
                responses.append(pred_solution)
    return questions, responses, observations, standard_answers




# 方法用于得到encode编码
def process_data_for_trainer(data_words):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model_inputs = []
    tokenizer.pad_token = tokenizer.eos_token

    for i in range(len(data_words)):
        input_ids = tokenizer.encode(data_words[i], max_length=256, padding='max_length', truncation=True,
                                     return_tensors="pt")
        input_ids = input_ids[0]
        padding_mask = input_ids != tokenizer.pad_token_id

        model_inputs.append(((input_ids * padding_mask)))
    return model_inputs
def process_data_for_trainer_1000(data_words):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model_inputs = []
    tokenizer.pad_token = tokenizer.eos_token

    for i in range(len(data_words)):
        # input_ids = tokenizer.encode(data_words[i], return_tensors="pt")
        input_ids = tokenizer.encode(data_words[i], max_length=1000, padding='max_length', truncation=True,
                                     return_tensors="pt")
        input_ids = input_ids[0]
        # 创建掩码张量，其中填充的位置为 0，非填充的位置为 1
        padding_mask = input_ids != tokenizer.pad_token_id

        # 按位点乘掩码
        # model_inputs.append(((input_ids * padding_mask).expand(150, 150)))
        model_inputs.append(((input_ids * padding_mask)))
    return model_inputs


# 方法用于得到encode编码
def pro_data(question, response, observation, standard_answer, state):
    observation_zong, response_zong = [], []
    questions = process_data_for_trainer(question)
    # observation_zong = ['\n'.join(observation)]
    # response_zong = ['\n'.join(response)]
    observations = process_data_for_trainer(observation)
    responses = process_data_for_trainer(response)
    standard_answers = process_data_for_trainer(standard_answer)
    question.append(observation[0])
    question.append(observation[1])
    question.append(observation[2])
    state = question
    states = process_data_for_trainer_1000(['\n'.join(state)])
    return np.array(questions[0]), np.array(responses), np.array(observations), np.array(standard_answers[0]), np.array(states[0])

def construct_message(agents, question, idx,i):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct."
                                           " Please reiterate your answer, "
                                           "with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    role_prompt_model = ['Using the opinion of other agents as additional advice.',
                          'Please stick to your own point of view for debate.',
                          'Please focus on the answers from other agents as a reference.']


    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    role_prompt_model = role_prompt_model[i]
    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? {}\n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(role_prompt_model, question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def extract_content_after_inst(text):
    # 查找 '[/INST]' 标记的位置
    text = '\n'.join(text)
    inst_index = text.find('[/INST]')

    if inst_index != -1:
        # 获取 '[/INST]' 标记之后的内容
        content = text[inst_index + 7:]
        return {"role": "assistant", "content": content}
    else:
        # 如果没有找到标记，返回空字符串
        return ""

def truncate_sequence(input_ids, history_max_len, tokenizer):
    if input_ids.shape[1] > history_max_len:
        input_ids = input_ids[:, :history_max_len]
    return input_ids


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def build_dataset(query_dataset, input_min_text_length=2, input_max_text_length=8):

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        ds = query_dataset.filter(lambda x: len(x["question"]) > 200, batched=False)


        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["question"])
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch", output_all_columns=True)

        return ds
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model1 = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16,
    eos_token_id=2,
    pad_token_id=2,
    offload_state_dict=False,
)
# model1.gradient_checkpointing_enable()
model2 = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16,
    eos_token_id=2,
    pad_token_id=2,
    offload_state_dict=False,
)
# model2.gradient_checkpointing_enable()
model3 = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16,
    eos_token_id=2,
    pad_token_id=2,
    offload_state_dict=False,
)
# model3.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
history_token_ids = tokenizer('<s>', return_tensors="pt").input_ids

model1_lora = get_peft_model(model1, lora_config)
model2_lora = get_peft_model(model2, lora_config)
model3_lora = get_peft_model(model3, lora_config)

if __name__ == "__main__":
    args = get_common_args()
    args = qmix_args(args)

    args.model_lora = model1_lora, model2_lora, model3_lora
    env = RandomEnv(args)
    agents = Agents(args)
    worker = RolloutWorker(env, agents, args)
    buffer = ReplayBuffer(args)

    agents_num = 3
    rounds = 2

    model = model1, model2, model3

    generated_description = {}

    questions = read_jsonl("")

    for epoch in range(args.n_epoch):
        random.shuffle(questions)
        print('train epoch {}'.format(epoch))
        # 清空存储的剧集数据
        episodes = []
        for episode_idx in tqdm(range(args.n_episodes)):
            data = questions[episode_idx]
            question = data['question']
            answer = data['answer']
            agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning.
             Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your 
             response. """.format(question)}] for agent_num in range(agents_num)]

            for round in range(rounds):
                for i, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1:]
                        message = construct_message(agent_contexts_other, question, 2 * round - 1, i)
                        agent_context.append(message)

                    completion = format_tokens(agent_context, tokenizer)

                    completion = torch.tensor(completion).to(device)

                    completion = truncate_sequence(completion, history_max_len, tokenizer)
                    with torch.no_grad():
                        outputs = model[i].generate(input_ids=completion, max_new_tokens=max_new_tokens,
                                                    do_sample=True,
                                                    top_p=top_p,
                                                    temperature=temperature,
                                                    repetition_penalty=repetition_penalty,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    pad_token_id=tokenizer.eos_token_id,
                                                    attention_mask=torch.ones(completion.shape,
                                                                              dtype=torch.long,
                                                                              device=device)
                                                    )
                    response = tokenizer.batch_decode(outputs)

                    assistant_message = extract_content_after_inst(response)

                    agent_context.append(assistant_message)

            generated_description[question] = (agent_contexts, answer)

            questions_q, responses_q, observations_q, standard_answers_q = response_saved_data(
                generated_description)  # 得到文本
            states_q = []
            questions_q, responses_q, observations_q, standard_answers_q, states_q = pro_data(questions_q,
                                                                                              responses_q,
                                                                                              observations_q,
                                                                                              standard_answers_q,
                                                                                              states_q)  # 得到encode编码

            save_path = args.result_dir + '/' + args.alg

            args.standard_answer = standard_answers_q
            args.state = states_q
            args.observation = observations_q
            args.response = responses_q
            # args.round = rounds

            episode, _ = worker.generate_episode(episode_idx)
            episodes.append(episode)
        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                try:
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
                except Exception as e:
                    print('episode_batch[key].shape=', episode_batch[key].shape)
                    print('episode[key].shape=', np.array([episode[key]]).shape)
                    print('错误：', e)
        buffer.store_episode(episode_batch)

        # 从缓冲区随机取样进行训练
        for train_step in range(args.train_steps):
            mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
            agents.train(mini_batch, train_steps)
            train_steps += 1
