import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
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

win_rates = []
# 初始化胜率列表
episode_rewards = []
# 初始化每集奖励列表
train_steps = 0
# 初始化训练步骤计数器





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




device = torch.device("cuda:0")
max_new_tokens = 500
history_max_len = 2000
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.2
model_id = ""

model1 =  AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16,
    eos_token_id=2,
    pad_token_id=2,
)
model2 =  AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16,
    eos_token_id=2,
    pad_token_id=2,
)
model3 =  AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16,
    eos_token_id=2,
    pad_token_id=2,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)



history_token_ids = tokenizer('<s>', return_tensors="pt").input_ids

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
if __name__ == "__main__":
    agents_num = 3
    rounds = 2
    random.seed(0)
    model = model1, model2, model3


    generated_description = {}

    # questions = read_jsonl("/data01/lihaoran/lhr/maq_ppo/dataset/ASDiv/ASDiv.json")
    # load arguments from terminal
    args_extra = arg_parser()
    print('*****************************')
    print(args_extra)
    print('*****************************')
    # print(f"API_KEY: {API_KEY}")
    set_random_seed(args_extra.random_seed)
    # load dataset
    dataloader = create_dataloader(args_extra)

    epochs = 10
    for epoch_w in range(epochs):
        random.shuffle(dataloader)
        # print('数据长度为：',len(questions))
        args = get_common_args()
        # 获取公共参数(env
        args = qmix_args(args)
        # 使用qmix_args函数修改参数(env

        ppo_data = load_dataset('/data01/lihaoran/lhr/maq_ppo/saved_data/ppo_gsm', split="train")


        for data in tqdm(dataloader[:100]):

            # ppo_data=ppo_data.shuffle().train_test_split(test_size=0.005)['test']
            # ppo_data=build_dataset(ppo_data)
            # ppo = PPO(ppo_data)
            # ppo.run()

            question = data['question']
            answer = data['answer']

            agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning.
             Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your 
             response. """.format(question)}] for agent_num in range(agents_num)]

            if os.path.exists('/data01/lihaoran/lhr/maq_ppo/src/qmix_tuning_0521/my_ppo_model/pytorch_model.bin'):
                torch.load('/data01/lihaoran/lhr/maq_ppo/src/qmix_tuning_0521/my_ppo_model/pytorch_model.bin')
            else:
                print('没有已保存的ppo模型')

            for round in range(rounds):
                for i, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1:]
                        message = construct_message(agent_contexts_other, question, 2 * round - 1,i)
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
                                                    attention_mask=torch.ones(completion.shape,dtype=torch.long,device=device)
                                                    )
                    # try:

                    # except Exception as e:
                    #     print("error:处理的文本为", completion)
                    response = tokenizer.batch_decode(outputs)

                    assistant_message = extract_content_after_inst(response)

                    agent_context.append(assistant_message)

            generated_description[question] = (agent_contexts, answer)

            questions_q, responses_q, observations_q, standard_answers_q = response_saved_data(
                generated_description)  # 得到文本
            states_q = []
            questions_q, responses_q, observations_q, standard_answers_q, states_q = pro_data(questions_q, responses_q,
                                                                                              observations_q,
                                                                                              standard_answers_q,
                                                                                              states_q)  # 得到encode编码

            save_path = args.result_dir + '/' + args.alg
            # 设置保存结果的路径

            args.standard_answer = standard_answers_q
            args.state = states_q
            args.observation = observations_q
            args.response = responses_q

            env = RandomEnv(args)
            # 创建一个随机环境实例，args参数用于配置环境(env
            agents = Agents(args)
            # 创建智能体实例，args参数用于配置智能体的行为(agent
            worker = RolloutWorker(env, agents, args)
            # 创建RolloutWorker实例，用于在环境中执行剧集
            buffer = ReplayBuffer(args)

            for epoch in range(args.n_epoch):
                # 打印当前运行和训练轮次
                # print('Run {}, train epoch {}'.format(epoch_w, epoch))
                # 清空存储的剧集数据
                episodes = []
                # 生成剧集数据
                for episode_idx in range(args.n_episodes):
                    episode, _ = worker.generate_episode(episode_idx)
                    episodes.append(episode)
                # 取第一个剧集数据进行处理
                episode_batch = episodes[0]
                episodes.pop(0)
                # 合并其他剧集数据
                for episode in episodes:
                    for key in episode_batch.keys():
                        try:
                            episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
                        except Exception as e:
                            print('episode_batch[key].shape=', episode_batch[key].shape)
                            print('episode[key].shape=', np.array([episode[key]]).shape)
                            print('错误：', e)
                    # 将剧集数据存储到缓冲区
                # buffer.store_episode(episode_batch)

                # 从缓冲区随机取样进行训练
                for train_step in range(args.train_steps):
                    # mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
                    agents.train(episode_batch, train_steps)
                    train_steps += 1
                # torch.save(agents[0].state_dict(), f"model_{epoch}.pth")

        try:
            # 定义文件名格式
            filename_format = "svamp_{}_{}_{}.json"
            # 获取当前时间
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d%H%M%S")  # 格式化为YYYYMMDDHHMMSS
            # 格式化文件名，替换{}中的内容
            filename = filename_format.format(agents_num, rounds, timestamp)
            # 打开文件准备写入
            with open("/data01/lihaoran/lhr/maq_ppo/saved_data/gsm/" + filename, 'w') as file:
                # 将描述序列化为JSON格式并写入文件
                json.dump(generated_description, file, indent=4)
                print('创建成功',str(file))
            print(f"JSON file '{filename}' has been created.")
            Eval_gsm = Eval_Gsm("/data01/lihaoran/lhr/maq_ppo/saved_data/gsm/" + filename)
            accuracies = Eval_gsm.eval_gsm()
            # print(accuracies)
            if accuracies >= 0.35:
                with open("/data01/lihaoran/lhr/maq_ppo/saved_data/gsm/" + filename, 'a') as file:
                    # 将描述序列化为JSON格式并写入文件
                    file.write('\naccuracies:')
                    file.write(str(accuracies))
                print('accuracies>=35')
                break
        except Exception as e:
            print("An error occurred while saving the file:", e)

        # ppo=PPO('/data01/lihaoran/lhr/maq_ppo/saved_data/ppo_gsm')
        # ppo.run()
        # if os.path.exists('./my_ppo_model/ppo.pth'):
        #     model.load_state_dict(torch.load('./my_ppo_model/ppo.pth'))
        # else:
        #     print('没有已保存的ppo模型')
