#coding:utf-8-*-
import torch
import bitsandbytes as bnb
import numpy as np
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )
import torch
class RewardFunction:
    def __init__(self, standard_answer, reward_scale_rate=20):
        self.standard_answer = standard_answer
        self.reward_scale_rate = reward_scale_rate
        self.previous_answers = {}
        self.previous_scores = {}

    def calculate_reward(self, current_answers, current_scores, debug=False):
        """
        Calculate the reward for the current round of answers given by multiple agents.
        The reward is based on the consistency and the degree of contribution to the answer.

        :param current_answers: A dictionary of agent_id to answer strings.
        :param current_scores: A dictionary of agent_id to score values.
        :param debug: A flag to enable debugging features.
        :return: A dictionary of agent_id to rewards.
        """
        rewards = {}
        current_reward = 0

        # Check consistency
        for agent_id, answer in current_answers.items():
            if agent_id not in self.previous_answers:
                # First answer, no comparison possible
                rewards[agent_id] = 0
                continue

            previous_answer = self.previous_answers[agent_id]
            consistency_reward = self.calculate_consistency_reward(answer, previous_answer)
            rewards[agent_id] += consistency_reward
            current_reward += consistency_reward

        # Check contribution degree
        for agent_id, score in current_scores.items():
            contribution_reward = self.calculate_contribution_reward(score, self.standard_answer)
            rewards[agent_id] += contribution_reward
            current_reward += contribution_reward

        # Scale the reward based on the improvement
        improvement_reward = self.calculate_improvement_reward(current_reward, debug)
        for agent_id in current_answers:
            rewards[agent_id] += improvement_reward

        # Update previous answers and scores
        self.previous_answers = current_answers
        self.previous_scores = current_scores

        return rewards

    def calculate_consistency_reward(self, current_answer, previous_answer):
        # This function should calculate the reward based on the consistency of answers.
        # For simplicity, we can assume a simple heuristic: the more similar the answers,
        # the higher the reward.
        current_answer = [current_answer[i:i + 3] for i in range(0, len(current_answer), 3)]
        similarity_scores = []
        # index = 0
        for _current_answer in current_answer:
            for i in range(len(_current_answer)):
                # similarity_score = self.calculate_reward(_current_answer[i], previous_answer[index])
                similarity_score = self.calculate_reward(_current_answer[i], previous_answer)
                similarity_scores.append(similarity_score)
            # index += 1
        return similarity_scores

    # 定义一个奖励函数，它根据生成回答和参考回答的相似度来计算奖励值
    def calculate_reward(self, current_answer, previous_answer):

        similarity = torch.cosine_similarity(torch.tensor(current_answer).float().view(1, -1), torch.tensor(previous_answer).float().view(1, -1))


        reward = torch.clamp(similarity, min=-1.0, max=1.0)

        return reward.item()



    def calculate_contribution_reward(self, current_score, standard_answer):
        # This function should calculate the reward based on the contribution of the answer.
        # For simplicity, we can assume a simple heuristic: the more similar the answer to
        # the standard answer, the higher the reward.
        similarity_score = self.calculate_similarity_score(current_score, standard_answer)
        return similarity_score

    def calculate_improvement_reward(self, current_reward, debug=False):
        # This function should calculate the reward based on the improvement over the previous round.
        # In a simple version, we can just return the reward scale rate if debug is False,
        # or provide a more sophisticated calculation if debug is True.
        if debug:
            # Placeholder for sophisticated debug reward calculation
            return 0
        else:
            return self.reward_scale_rate

    def calculate_similarity_score(self, current_answer, previous_answer):
        # This function should calculate a score representing the similarity between two answers.
        # For simplicity, we can use the difference in their lengths as a proxy for similarity.
        # A lower difference in lengths could imply a higher reward.
        length_difference = abs(len(current_answer) - len(previous_answer))
        similarity_score = 1 / (1 + length_difference)
        return similarity_score
