import numpy as np
import os
from tqdm import tqdm
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
sns.set_theme()
signal_mapping = {0: 'incorrect', 1:'correct'}


def get_probs_from_LLM(task_str: str, candidate_set: list[str]) -> Tuple[np.array, np.array]:
    """
    Using the task string and candidate answers as prompt
    Generate normalized probability predicted by LLM

    Return: Generator probs, Discriminator probs
    """
    candidate_size = len(candidate_set)
    # gen: P(y|v), 2->Y
    gen_probs = np.empty([2, candidate_size], dtype=float)
    # dis: P(v|y), Y->2
    dis_probs = np.empty([candidate_size, 2], dtype=float)

    # TODO: get probs from LLM logits
    gen_probs[0][0], gen_probs[1][-1] = 0.9, 0.9
    gen_probs[0][1:], gen_probs[1][:-1] = 0.1 / \
        (candidate_size - 1),  0.1 / (candidate_size - 1)
    tmp = np.random.random(size=dis_probs.shape)
    dis_probs = tmp/tmp.sum(axis=-1, keepdims=True)
    dis_probs[0], dis_probs[-1] = [0.9, 0.1], [0.5, 0.5]

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
    """

    def __init__(self, task_str: str, candidate_set: list[str]):
        # save args
        self.task_str = task_str
        self.candidate_set = candidate_set
        self.candidate_size = len(candidate_set)
        # learning param
        self.eta_g = self.eta_d = 0.1
        self.lam_g = self.lam_d = 0.1
        # initialize probs
        self.init_gen, self.init_dis = get_probs_from_LLM(
            task_str, candidate_set)
        # current probs
        self.gen, self.dis = np.copy(self.init_gen), np.copy(self.init_dis)
        self.Q_gen, self.Q_dis = np.zeros_like(
            self.gen), np.zeros_like(self.dis)

    def naive_run(self, max_iter=5000):
        llog_init_gen, llog_init_dis = self.lam_g*np.log(self.init_gen), self.lam_d*np.log(self.init_dis)
        for t in tqdm(range(1, max_iter+1)):
            ## run the game
            # 1. nature selects v
            signal = np.random.randint(0, 2)
            # 2. generator generates y
            y = np.random.choice(self.candidate_size, p=self.gen[signal])
            ## update weights
            # 1. Q value, perform incremental update
            self.Q_gen[signal] += 1/2/t*self.dis[y, signal]
            self.Q_dis[y] += 1/2/t*self.gen[signal, y]
            # 2. PiKL update
            rate_gen, rate_dis = 1/(self.eta_g*t) + self.lam_g, 1/(self.eta_d*t)+self.lam_d
            next_gen = np.exp((self.Q_gen + llog_init_gen) / rate_gen)
            next_dis = np.exp((self.Q_dis + llog_init_dis) / rate_dis)
            self.gen = next_gen / next_gen.sum(axis=-1, keepdims=True)
            self.dis = next_dis / next_dis.sum(axis=-1, keepdims=True)

    def plot_policy(self, filename='fig.pdf'):
        dir_name = 'figs'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig, axs = plt.subplots(1,2, figsize=(10,4))
        sns.heatmap(self.gen, yticklabels=signal_mapping.values(),xticklabels=self.candidate_set,
                    cmap='crest', annot=True, linewidth=.5, ax=axs[0])
        sns.heatmap(self.dis,xticklabels=signal_mapping.values(),yticklabels=self.candidate_set,
        cmap='crest', annot=True, linewidth=.5, ax=axs[1])
        axs[0].set_title('Generator')
        axs[1].set_title('Discriminator')
        fig.suptitle(f'Q:{self.task_str}, Answers:{" ".join(self.candidate_set)}')
        plt.tight_layout()
        plt.savefig(os.path.join(dir_name, filename))


if __name__ == '__main__':
    
    task_str = 'Is it possible that he acted it? Did he act it?'
    candidate_set = ['A.yes', 'B.maybe', 'C.obtuse angle']
    game = Consensus_Game(task_str=task_str, candidate_set=candidate_set)

    game.plot_policy('init_policy.pdf')

    game.naive_run()

    game.plot_policy('end_policy.pdf')