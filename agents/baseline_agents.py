import abc

import numpy as np
import random

from agents.Agent import Agent
from mygym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment

from agents.value_approximators.baseline_nets import Net
from agents.value_approximators.Nets import Params, LSTM, DNN

#from torchsummary import summary

from copy import deepcopy


class RandomAgent(Agent):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
    ):
        super().__init__(learn_env, valid_env, False)

    def replay(self):
        pass

    def get_action(self, state: np.ndarray) -> int:
        return self.action_space.sample()

    def get_name(self):
        prefix = 'InventoryDriven' if not self.learn_env.market_order_clearing else ''
        return prefix+"RandomAgent"


class FixedActionAgent(Agent):
    def __init__(
            self,
            fixed_action: int = None,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None
    ):
        super().__init__(learn_env, valid_env, False)
        self.fixed_action = fixed_action

    def replay(self):
        pass

    def get_action(self, state: np.ndarray) -> int:
        return self.fixed_action

    def get_name(self) -> str:
        prefix = 'InventoryDriven' if not self.learn_env.market_order_clearing else ''
        return prefix+f"FixedAction_{self.fixed_action}"


class BaseDQN(Agent):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
            episodes: int = 1000,
            epsilon: float = 0.99,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.99,
            gamma: float = 0.97,
            batch_size: int = 512,
            type_algo: str = 'target',
            tau: float = 0.01
    ):
        super().__init__(learn_env, valid_env, True, episodes, epsilon, epsilon_min, epsilon_decay, gamma, batch_size)
        self.type_algo = type_algo
        self.tau = tau
        #self.replay_steps = 0
        self._set_seed_rand()

    def _set_seed_rand(self):
        random.seed(self.seed)

    @abc.abstractmethod
    def _compute_fit(self, state: np.ndarray, target, mask: np.ndarray = None):
        pass

    @abc.abstractmethod
    def _compute_prediction(self, model, state: np.ndarray, idmax: bool):
        pass

    def get_action(self, state: np.ndarray):
        """
        optimal policy is defined trivially as: when the agent is in state s, it will select the action with the highest
        value for that state
        """
        return self._compute_prediction(self.model, state, idmax=True)

    def _set_target_model(self):
        self.target_model = deepcopy(self.model)
        for param in self.target_model.model.parameters():
            param.requires_grad = False

    def replay(self):
        """
        Method to retrain the DQN model based on batches of memorized experiences
        Updating the policy function Q regularly, improve the learning considerably
        """
        batch = list(map(np.array, zip(*random.sample(self.memory, self.batch_size))))
        states, actions, rewards, next_states, dones = batch
        """
        approximate Q-Value(target) should be close to the reward the agent gets after playing action a in state s
        plus the future discounted value of playing optimally from then on
        """
        if self.type_algo == 'target':
            actions_next_step = np.argmax(self._compute_prediction(self.model, next_states, idmax=False).cpu().numpy(), axis=1)
            Q_values_target_next_step = self._compute_prediction(self.target_model, next_states, idmax=False).cpu().numpy()[range(len(actions)), actions_next_step]
            rewards += (1 - dones) * self.gamma * Q_values_target_next_step
        else:
            rewards += (1 - dones) * self.gamma * np.amax(self._compute_prediction(self.model, next_states, idmax=False).cpu().numpy(),
                                                          axis=1)

        all_Q_values = self._compute_prediction(self.model, states, idmax=False).cpu().numpy()
        all_Q_values_target = all_Q_values.copy()
        all_Q_values_target[range(len(actions)), actions] = rewards
        self._compute_fit(states, all_Q_values_target)

        if self.type_algo == 'target':
            target_net_state_dict = self.target_model.model.state_dict()
            policy_net_state_dict = self.model.model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.target_model.model.load_state_dict(target_net_state_dict)


class DnnAgent(BaseDQN):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
            hidden_dim: int = 256,
            n_hidden: int = 1,
            lr: float = 0.001,
            dropout: float = 0.1

    ):
        super().__init__(learn_env, valid_env)
        self._set_model(hidden_dim, n_hidden, lr, dropout)
        self._set_target_model()

    def get_name(self):
        prefix1 = 'InventoryDriven' if not self.learn_env.market_order_clearing else ''
        if self.learn_env.per_step_reward_function.inventory_aversion == 0.1:
            prefix2 = 'SmallDampen'
        elif self.learn_env.per_step_reward_function.inventory_aversion == 0:
            prefix2 = 'NoDampen'
        else:
            prefix2 = 'HighDampen'
        return prefix1+prefix2+"DnnAgent"

    def _set_model(self, hidden_dim: int, n_hidden: int, lr: float, dropout: float):
        if n_hidden == 0: dropout = 0
        assert (self.learn_env.n_lags_feature == 0)
        params = Params(input_dim=len(self.learn_env.features), hidden_dim=hidden_dim, n_hidden=n_hidden,
                        dropout=dropout, seed=self.seed)
        self.model = Net(DNN(params), lr=lr, name=self.get_name(), seed=self.seed)
        summary(self.model.model, (1, len(self.learn_env.features)))

    def _compute_fit(self, state: np.ndarray, target):
        self.model.fit(state, target)

    def _compute_prediction(self, model, state: np.ndarray, idmax: bool):
        return model.predict(state, idmax=idmax)


class LstmAgent(BaseDQN):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
            hidden_dim: int = 256,
            n_hidden: int = 1,
            lr: float = 0.001,
            dropout: float = 0.1

    ):
        super().__init__(learn_env, valid_env)
        self._set_model(hidden_dim, n_hidden, lr, dropout)
        self._set_target_model()

    def get_name(self):
        prefix1 = 'InventoryDriven' if not self.learn_env.market_order_clearing else ''
        if self.learn_env.per_step_reward_function.inventory_aversion == 0.1:
            prefix2 = 'SmallDampen'
        elif self.learn_env.per_step_reward_function.inventory_aversion == 0:
            prefix2 = 'NoDampen'
        else:
            prefix2 = 'HighDampen'
        return prefix1+prefix2+"LstmAgent"

    def _set_model(self, hidden_dim: int, n_hidden: int, lr: float, dropout: float):
        if n_hidden == 0: dropout = 0
        params = Params(input_dim=len(self.learn_env.features), hidden_dim=hidden_dim, n_hidden=n_hidden,
                        dropout=dropout, seed=self.seed)
        self.model = Net(LSTM(params), lr=lr, name=self.get_name(), seed=self.seed)
        summary(self.model.model, (self.learn_env.n_lags_feature, len(self.learn_env.features)))

    def _compute_fit(self, state: np.ndarray, target):
        self.model.fit(state, target)

    def _compute_prediction(self, model, state: np.ndarray, idmax: bool):
        return model.predict(state, idmax=idmax)
