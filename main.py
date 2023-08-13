from datetime import datetime
from agents.baseline_agents import FixedActionAgent, RandomAgent, LstmAgent, DnnAgent
from utils.utils import split_dates, envs_creator
from mygym.utils import plot_eval

if __name__ == '__main__':

    ticker = "MSFT"
    ticker_test = "GOOG"
    step_in_sec = 1
    date = datetime(2012, 6, 21)

    dates = split_dates(split=0.8, date=date, hour_start=10, hour_end=15.5, step_in_sec=step_in_sec)

    params = {
        'no_clearing': 1e10, 'clearing': 10000,
        'lags_dnn': 0, 'lags_lstm': 60,
        'small_damp': 0.1, 'high_damp': 0.7
        }

    pnls_msft, invs_msft = [], []
    pnls_goog, invs_goog = [], []

    train_env, eval_env = envs_creator(ticker, dates, step_in_sec, lags=params['lags_lstm'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['small_damp'])
    agent = LstmAgent(train_env, eval_env)
    pnl, inv = agent.learn()

    """
    SDLinearAgent: inventory-driven, η=0.1, layers=Linear()
    """
    train_env, eval_env = envs_creator(ticker, dates, step_in_sec, lags=params['lags_dnn'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['small_damp'])
    agent = DnnAgent(train_env, eval_env)
    pnl, inv = agent.learn()
    pnls_msft.append(pnl)
    invs_msft.append(inv)
    _, eval_env = envs_creator(ticker_test, dates, step_in_sec, lags=params['lags_dnn'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['small_damp'])
    pnl, inv = agent.evaluate(eval_env)
    pnls_goog.append(pnl)
    invs_goog.append(inv)

    """
    HDLinearAgent: inventory-driven, η=0.7, layers=Linear()
    """
    train_env, eval_env = envs_creator(ticker, dates, step_in_sec, lags=params['lags_dnn'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['high_damp'])
    agent = DnnAgent(train_env, eval_env)
    pnl, inv = agent.learn()
    pnls_msft.append(pnl)
    invs_msft.append(inv)
    _, eval_env = envs_creator(ticker_test, dates, step_in_sec, lags=params['lags_dnn'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['high_damp'])
    pnl, inv = agent.evaluate(eval_env)
    pnls_goog.append(pnl)
    invs_goog.append(inv)

    """
    SDLstmAgent: inventory-driven, η=0.1, layers=LSTM()
    """
    train_env, eval_env = envs_creator(ticker, dates, step_in_sec, lags=params['lags_lstm'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['small_damp'])
    agent = LstmAgent(train_env, eval_env)
    pnl, inv = agent.learn()
    pnls_msft.append(pnl)
    invs_msft.append(inv)
    _, eval_env = envs_creator(ticker_test, dates, step_in_sec, lags=params['lags_lstm'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['small_damp'])
    pnl, inv = agent.evaluate(eval_env)
    pnls_goog.append(pnl)
    invs_goog.append(inv)

    """
    HDLstmAgent: inventory-driven, η=0.7, layers=LSTM()
    """
    train_env, eval_env = envs_creator(ticker, dates, step_in_sec, lags=params['lags_lstm'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['high_damp'])
    agent = LstmAgent(train_env, eval_env)
    pnl, inv = agent.learn()
    pnls_msft.append(pnl)
    invs_msft.append(inv)
    _, eval_env = envs_creator(ticker_test, dates, step_in_sec, lags=params['lags_lstm'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['high_damp'])
    pnl, inv = agent.evaluate(eval_env)
    pnls_goog.append(pnl)
    invs_goog.append(inv)

    """
    MCLstmAgent: market clearing, η=0, layers=LSTM()
    """
    train_env, eval_env = envs_creator(ticker, dates, step_in_sec, lags=params['lags_lstm'],
                                       max_inv=params['clearing'], inventry_aversion=0)
    agent = LstmAgent(train_env, eval_env)
    pnl, inv = agent.learn()
    pnls_msft.append(pnl)
    invs_msft.append(inv)

    plot_eval(pnls_msft,
              invs_msft,
              pnls_goog,
              invs_goog)

    """
    RandomAgent
    """
    train_env, eval_env = envs_creator(ticker, dates, step_in_sec, lags=params['lags_dnn'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['small_damp'])
    agent = RandomAgent(train_env, eval_env)
    agent.learn()
    _, eval_env = envs_creator(ticker_test, dates, step_in_sec, lags=params['lags_dnn'],
                                       max_inv=params['no_clearing'], inventry_aversion=params['small_damp'])
    agent.evaluate(eval_env)

    """
    FixedAgent
    """
    for i in range(9):
        train_env, eval_env = envs_creator(ticker, dates, step_in_sec, lags=params['lags_dnn'],
                                           max_inv=params['no_clearing'], inventry_aversion=params['small_damp'])
        agent = FixedActionAgent(i, train_env, eval_env)
        agent.learn()
        _, eval_env = envs_creator(ticker_test, dates, step_in_sec, lags=params['lags_dnn'],
                                   max_inv=params['no_clearing'], inventry_aversion=params['small_damp'])
        agent.evaluate(eval_env)

    print('End')