from datetime import datetime, timedelta
import argparse
from helpers.main_helper import (
    add_env_args,
    get_env_configs,
)
from mygym.utils import env_creator
import pickle
import io
import torch


def split_dates(split: float = None, date: datetime = None, hour_start: float = None, hour_end: float = None,
                step_in_sec: float = None):
    start_train_date = date + timedelta(hours=hour_start) + timedelta(seconds=step_in_sec)
    end_test_date = date + timedelta(hours=hour_end)
    end_train_date = (start_train_date + (end_test_date - start_train_date) * split).replace(microsecond=0)
    start_test_date = end_train_date + timedelta(seconds=step_in_sec)
    return [start_train_date, end_train_date, start_test_date, end_test_date]


def envs_creator(ticker, dates, step_in_sec, lags, max_inv, inventry_aversion: float=0.1):
    parser = argparse.ArgumentParser(description="")
    add_env_args(parser, ticker, dates, step_in_sec, lags, max_inv, inventry_aversion)
    args = vars(parser.parse_args())
    train_env_config, eval_env_config = get_env_configs(args)
    train_env = env_creator(train_env_config)
    eval_env = env_creator(eval_env_config)
    return train_env, eval_env


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)