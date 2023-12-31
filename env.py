import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, List
import seaborn as sns
import matplotlib.pyplot as plt
from test_llama import *

np.random.seed(42)
sns.set_theme()
signal_mapping = {0: 'incorrect', 1: 'correct'}


def get_probs_from_LLM(task_str: str, candidate_text: List[str], candidate_id: List[str]) -> Tuple[np.array, np.array]:
    """
    Using the task string and candidate answers as prompt
    Generate normalized probability predicted by LLM

    Return: Generator probs, Discriminator probs
    """
    candidate_size = len(candidate_text)
    # gen: P(y|v), 2->Y
    gen_probs = np.empty([2, candidate_size], dtype=float)
    # dis: P(v|y), Y->2
    dis_probs = np.empty([candidate_size, 2], dtype=float)

    # use default path if no local file
    tokenizer, model = load_llama(path='D:/Codes/transformers/Llama-2-7b-hf')

    # generator
    for signal in range(2):
        is_correct = signal_mapping[signal] == 'correct'
        gen_probs[signal] = get_norm_probs(tokenizer, model, task_prompt=gen_task(
            task_str, candidate_text, candidate_id, correct=is_correct),
            candidate_set=candidate_id, verbose=False)

    # discriminator
    dis_candidate = ['False', 'True']
    for idx, ans_text in enumerate(candidate_text):
        dis_probs[idx] = get_norm_probs(tokenizer, model, task_prompt=dis_task(
            task_str, ans_text, incor_text=dis_candidate[0], cor_text=dis_candidate[1]),
            candidate_set=dis_candidate, verbose=False)

    return gen_probs, dis_probs


class Consensus_Game:
    """
    The consensus game: 2 player signaling game
    Given a text-based task x:
    1. Nature sample a correct/incorrect signal v (0/1)
    2. Generator generates answer by P(y|x,v)
    3. Discriminator evaluates the answer by P(·|x,y)

    What determines the game is
    1. task x
    2. candidates set Y

    Consensus game: https://arxiv.org/pdf/2310.09139.pdf
    PiKL: https://proceedings.mlr.press/v162/jacob22a/jacob22a.pdf
    """

    def __init__(self, task_str: str, candidate_text: List[str], candidate_id: List[str]):
        # save args
        self.task_str = task_str
        self.candidate_text = candidate_text
        self.candidate_id = candidate_id
        self.candidate_size = len(candidate_text)
        # learning param
        self.eta_g = self.eta_d = 0.1
        self.lam_g = self.lam_d = 0.1
        # initialize probs
        self.init_gen, self.init_dis = get_probs_from_LLM(
            task_str, candidate_text, candidate_id)
        # current probs
        self.gen = self.Q_gen = np.zeros_like(self.init_gen)
        self.dis = self.Q_dis = np.zeros_like(self.init_dis)

    def reset(self):
        self.gen = self.Q_gen = np.zeros_like(self.init_gen)
        self.dis = self.Q_dis = np.zeros_like(self.init_dis)

    def run(self, max_iter=5000):
        llog_init_gen, llog_init_dis = self.lam_g * \
            np.log(self.init_gen), self.lam_d*np.log(self.init_dis)
        for t in tqdm(range(1, max_iter+1)):
            ## set the policy according to Q
            rate_gen, rate_dis = 1/(self.eta_g*t) + \
                self.lam_g, 1/(self.eta_d*t) + self.lam_d
            next_gen = np.exp((self.Q_gen + llog_init_gen) / rate_gen)
            next_dis = np.exp((self.Q_dis + llog_init_dis) / rate_dis)
            self.gen = next_gen / next_gen.sum(axis=-1, keepdims=True)
            self.dis = next_dis / next_dis.sum(axis=-1, keepdims=True)
            ## run the game
            # 1. nature selects v
            signal = np.random.randint(0, 2)
            # 2. generator generates y
            y = np.random.choice(self.candidate_size, p=self.gen[signal])
            # 3. discriminatory evaluate
            v = np.random.choice(2, p=self.dis[y])
            utility = v == signal
            # dis_cfr = np.arange(2) == signal
            ## update Q, perform incremental update
            self.Q_gen[signal, y] += 1/t*(utility/2 - self.Q_gen[signal, y])
            self.Q_dis[y, signal] += 1/t*(utility/2 - self.Q_dis[y, signal])
            # self.Q_dis[y] += 1/2/t*dis_cfr
            if t % int(max_iter/5) == 1:
                self.plot_policy(f'iter={t}.png')

    def naive_run(self, max_iter=5000):
        llog_init_gen, llog_init_dis = self.lam_g * \
            np.log(self.init_gen), self.lam_d*np.log(self.init_dis)
        # set initial policy
        self.gen, self.dis = self.init_gen.copy(), self.init_dis.copy()
        for t in tqdm(range(1, max_iter+1)):
            ## randomize signal
            signal = np.random.randint(0, 2)
            ## update Q using probs directly
            self.Q_gen[signal] += 1/t * \
                (self.dis[:, signal]/2 - self.Q_gen[signal])
            self.Q_dis[:, signal] += 1/t * \
                (self.gen[signal]/2 - self.Q_dis[:, signal])
            ## set the policy according to Q
            rate_gen, rate_dis = 1/(self.eta_g*t) + \
                self.lam_g, 1/(self.eta_d*t) + self.lam_d
            next_gen = np.exp((self.Q_gen + llog_init_gen) / rate_gen)
            next_dis = np.exp((self.Q_dis + llog_init_dis) / rate_dis)
            self.gen = next_gen / next_gen.sum(axis=-1, keepdims=True)
            self.dis = next_dis / next_dis.sum(axis=-1, keepdims=True)
            if t % int(max_iter/5) == 1:
                self.plot_policy(f'naive_iter={t}.png')

    def plot_policy(self, filename='fig.png', init_policy=False):
        if init_policy:
            gen, dis = self.init_gen, self.init_dis
        else:
            gen, dis = self.gen, self.dis
        dir_name = 'figs'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(gen, yticklabels=signal_mapping.values(), xticklabels=self.candidate_id,
                    cmap='crest', annot=True, linewidth=.5, ax=axs[0])
        sns.heatmap(dis, xticklabels=signal_mapping.values(), yticklabels=self.candidate_id,
                    cmap='crest', annot=True, linewidth=.5, ax=axs[1])
        axs[0].set_title('Generator')
        axs[1].set_title('Discriminator')
        fig.suptitle(
            f'Q:"{self.task_str}", answers: {" ".join(["{}. {}".format(x,y) for x,y in zip(self.candidate_id, self.candidate_text)])}')
        plt.tight_layout()
        plt.savefig(os.path.join(dir_name, filename))


if __name__ == '__main__':

    task_str = 'Where was Barack Obama born?'
    candidate_text = ['Chicago', 'honolulu', 'Nairobi', 'NYC']
    candidate_id = ['A', 'B', 'C', 'D']
    game = Consensus_Game(
        task_str=task_str, candidate_text=candidate_text, candidate_id=candidate_id)

    game.plot_policy('init_policy.png', init_policy=True)
    game.run()
    game.plot_policy('end_policy.png')

    game.reset()
    game.naive_run()
    game.plot_policy('naive_end_policy.png')
