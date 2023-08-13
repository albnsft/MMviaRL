import abc
import numpy as np
from mygym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from mygym.utils import done_inf, plot_per_episode, plot_final, info_eval
from collections import deque
from pylab import plt, mpl
from copy import deepcopy
import os
import pickle
from utils.utils import CPU_Unpickler

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'


class ActionSpace:
    """
    Agent's exploration policy
    Simple uniform random policy between each action possible
    """

    def __init__(self, n):
        self.n = n

    def sample(self) -> int:
        return np.random.randint(0, self.n)


class Agent(metaclass=abc.ABCMeta):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            test_env: HistoricalOrderbookEnvironment = None,
            learning_agent: bool = None,
            episodes: int = None,
            epsilon: float = None,
            epsilon_min: float = None,
            epsilon_decay: float = None,
            gamma: float = None,
            batch_size: int = None,
            seed: int = 42
    ):
        self.learn_env = learn_env
        self.test_env = test_env
        self.learning_agent = learning_agent
        self.episodes = episodes
        self.seed = seed
        self._set_learning_args(epsilon, epsilon_min, epsilon_decay, gamma, batch_size)
        self.num_actions = 9
        self.step_info_per_episode = dict(map(lambda i: (i, None), range(1, self.episodes + 1)))
        self.step_info_per_eval_episode = deepcopy(self.step_info_per_episode)
        self.done_info = {'nd_pnl': [], 'map': [], 'aum': [], 'depth': []}
        self.done_info_eval = deepcopy(self.done_info)
        self.len_learn = None
        self.len_eval = None
        self.total_steps = 0

    def _set_seed_np(self):
        np.random.seed(self.seed)

    def _set_learning_args(self, epsilon: float, epsilon_min: float, epsilon_decay: float, gamma: float,
                           batch_size: int):
        self._set_seed_np()
        if self.learning_agent:
            self.epsilon = epsilon  # Initial exploration rate
            self.epsilon_min = epsilon_min  # Minimum exploration rate
            self.epsilon_decay = epsilon_decay  # Decay rate for exploration rate, interval must be a pos function of the nb of xp
            self.gamma = gamma  # Discount factor for delayed reward
            self.batch_size = batch_size  # Batch size for replay
            self.memory = deque(maxlen=int(10e5))  # deque collection for limited history to train agent
            self.test_env.base_threshold = 1
        else:
            self.episodes = 1
            self.learn_env.base_threshold = 1
            self.test_env.base_threshold = 1
            self.epsilon = 0
        if self.learn_env.max_inventory > 10000:
            self.learn_env.market_order_fraction_of_inventory, self.test_env.market_order_fraction_of_inventory = 0, 0  #no market order clearing
            self.learn_env.market_order_clearing, self.test_env.market_order_clearing = False, False

    @property
    def actions(self):
        return list(range(self.num_actions))

    @property
    def action_space(self):
        return ActionSpace(len(self.actions))

    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    def _greedy_policy(self, state: np.ndarray):
        self.total_steps += 1
        if np.random.random() <= self.epsilon:
            return self.action_space.sample()
        return self.get_action(state)

    def _play_one_step(self, state: np.ndarray):
        state = state.copy()
        action = self._greedy_policy(state) if self.learning_agent else self.get_action(state)
        next_state, reward, done, info = self.learn_env.step(action)
        if self.learning_agent:
            self.memory.append(
                [state, action, reward, next_state, done])
        return next_state, done

    @abc.abstractmethod
    def replay(self):
        pass

    def learn(self):
        print(f'****************************************{self.get_name()}****************************************')
        last_ep = self._set_args()
        for episode in range(last_ep, self.episodes + 1):
            state = self.learn_env.reset(random_time=True)
            self.len_learn = (self.learn_env.terminal_time - self.learn_env.state.now_is) / self.learn_env.step_size
            while self.learn_env.end_of_trading >= self.learn_env.state.now_is:
                state, done = self._play_one_step(state)
                if self.learning_agent and len(self.memory) > self.batch_size and (self.total_steps) % 100 == 0:
                    self.replay()
                if done:
                    self.step_info_per_episode[episode] = self.learn_env.info_calculator
                    self._compute_done(self.step_info_per_episode, episode, self.done_info)
                    break
            self._evaluate(episode)
            if self.learning_agent:
                self._save_args(episode)
                if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
            if self.learning_agent and episode >= 50 and (episode - 1) % 10 == 0:
                plot_final(self.done_info_eval, self.learn_env.ticker, self.get_name(),
                           self.learn_env.step_size, self.learn_env.market_order_fraction_of_inventory,
                           self.learn_env.per_step_reward_function)
        if self.learning_agent:
            best_info_eval = done_inf(self.done_info_eval).sort_values('aum', ascending=False).iloc[:5]
            print(f'Top 5 episodes on testing according to maximal AUM:\n  {best_info_eval.to_string()}')
            plot_final(self.done_info_eval, self.learn_env.ticker, self.get_name(),
                       self.learn_env.step_size, self.learn_env.market_order_fraction_of_inventory,
                       self.learn_env.per_step_reward_function)
            plot_per_episode(self.test_env.ticker, self.get_name(),
                             self.test_env.step_size, self.test_env.market_order_fraction_of_inventory,
                             self.test_env.per_step_reward_function, None,
                             self.step_info_per_eval_episode, done_inf(self.done_info_eval).sort_values('aum', ascending=False).index[0], None, self.done_info_eval)
            self._set_best_ep()
        pnl, inv = self.evaluate(self.test_env)
        return pnl, inv

    def _evaluate(self, episode: int):
        """
        Method to validate the performance of the DQL agent.
        only relies on the exploitation of the currently optimal policy
        """
        state = self.test_env.reset(random_time=False)
        self.len_eval = (self.test_env.end_of_trading - self.test_env.state.now_is) / self.test_env.step_size
        while self.test_env.end_of_trading >= self.test_env.state.now_is:
            action = self.get_action(state)
            state, reward, done, info = self.test_env.step(action)
            if done:
                self.step_info_per_eval_episode[episode] = self.test_env.info_calculator
                self._compute_done(self.step_info_per_eval_episode, episode, self.done_info_eval)
                break

    def _compute_done(self, info, episode: int, done_info: dict):
        info = info[episode]
        done_info['nd_pnl'].append(info.nd_pnl)
        done_info['map'].append(info.map)
        done_info['aum'].append(info.aum)
        bar = len(info.inventories)
        done_info['depth'].append(bar)
        if (episode - 1) % 10 == 0:
            templ = '\nepisode: {:2d}/{} | bar: {:2d}/{} | epsilon: {:5.2f}\n'
            templ += 'normalised pnl: {:5.2f} | mean abs position: {:5.2f}\n'
            templ += 'asset under management: {:5.2f} | success: {} \n'
            if done_info is self.done_info:
                print(50 * '*')
                print(f'           Training of {self.get_name()}      ')
                print(f'    Start of trading: {self.learn_env.terminal_time - self.learn_env.episode_length} ')
                if bar > round(self.len_learn): bar = round(self.len_learn)
                success = True if bar == round(self.len_learn) else False
                print(templ.format(episode, self.episodes, bar, round(self.len_learn), self.epsilon,
                                   info.nd_pnl, info.map, info.aum, success))
            else:
                print(f'          Evaluation of {self.get_name()}      ')
                print(f'    Start of trading: {self.test_env.start_of_trading} ')
                if bar > round(self.len_eval): bar = round(self.len_eval)
                success = True if bar == round(self.len_eval) else False
                print(templ.format(episode, self.episodes, bar, round(self.len_eval), self.epsilon,
                                   info.nd_pnl, info.map, info.aum, success))
                print(50 * '*')

    def _get_path(self, episode: str = None):
        ticker = self.learn_env.ticker
        base = 'agents'
        agent = os.path.join(base, 'savings', self.get_name(), ticker)
        if not os.path.exists(agent):
            os.makedirs(agent)
        return os.path.join(agent, 'Episode'+str(episode))

    def _save_args(self, episode: int = None):
        best_ep = done_inf(self.done_info_eval).sort_values('aum', ascending=False).index[0]
        path = self._get_path(episode)
        if best_ep == episode:
            #save best models
            best_path = path.replace('agents\\savings', 'best_agents').replace('Episode'+ str(episode), '')
            if not os.path.exists(best_path): os.makedirs(best_path)
            self.model.save_args(best_path)
            print(f'************** Best model yet (max AUM (=PnL) on test set) --> episode {episode} ************** ')
            plot_per_episode(self.learn_env.ticker, self.get_name(),
                             self.learn_env.step_size, self.learn_env.market_order_fraction_of_inventory,
                             self.learn_env.per_step_reward_function, self.step_info_per_episode,
                             self.step_info_per_eval_episode, episode, self.done_info, self.done_info_eval)
        if (episode - 1) % 10 == 0 or episode==self.episodes:
            #save agent every 10 steps to reload if job is killed
            self._delete_earlier_args(path, episode)
            self.model.save_args(path)
            self.target_model.save_args(path, 'target')
            model_agent, target_model_agent = self.model, self.target_model
            self.model, self.target_model = None, None
            with open(f'{path}_agent.pkl', "wb") as file:
                pickle.dump(self, file)
            self.model, self.target_model = model_agent, target_model_agent

    def _delete_earlier_args(self, path, episode):
        args_file = [d for d in os.listdir(path.replace('Episode' + str(episode), ''))]
        for file in args_file:
            delete_filename = os.path.join(path.replace('Episode' + str(episode), file))
            open(delete_filename, 'w').close()
            os.remove(delete_filename)

    def _set_args(self):
        if self.learning_agent:
            path = self._get_path()
            last_ep = self.model.set(path)
            self.target_model.set(path, prefix='target')
            if last_ep >= 1:
                mymodel, mytargetmodel = self.model, self.target_model
                print(f'Loading arguments from episode {last_ep} savings')
                try:
                    with open(path.replace('None', str(last_ep)) + '_agent.pkl', "rb") as file:
                        self.__dict__ = pickle.load(file).__dict__
                except:
                    with open(path.replace('None', str(last_ep)) + '_agent.pkl', "rb") as file:
                        self.__dict__ = CPU_Unpickler(file).load().__dict__
                if self.target_model is None: self.target_model = mytargetmodel
                self.model = mymodel
        else:
            last_ep = 0
        return last_ep + 1 #to not redo the same ep

    def _set_best_ep(self):
        """
        After learn() has been called over the 1000 episodes (self.episode), set the current model to the best one
        """
        if self.learning_agent:
            path = self._get_path()
            new_path = path.replace('agents\\savings', 'best_agents').replace('EpisodeNone', '')
            self.model.set(new_path)

    def evaluate(self, test_env: HistoricalOrderbookEnvironment):
        self.test_env = test_env
        self.test_env.threshold = 1
        self.step_info_per_eval_episode = {1: None}
        self.done_info_eval = {'nd_pnl': [], 'map': [], 'aum': [], 'depth': []}
        print('------------------------------------------------------------------------')
        print(f'----------------Evaluation of {self.test_env.ticker}------------------')
        print('------------------------------------------------------------------------')
        episode = 1
        self._evaluate(episode)
        plot_per_episode(self.test_env.ticker, self.get_name(),
                         self.test_env.step_size, self.test_env.market_order_fraction_of_inventory,
                         self.test_env.per_step_reward_function, None,
                         self.step_info_per_eval_episode, episode, None, self.done_info_eval)
        pnl, inv = info_eval(self.step_info_per_eval_episode, episode, self.get_name())
        return pnl, inv