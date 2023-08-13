from copy import deepcopy
from datetime import datetime


def add_env_args(parser, ticker: str, dates: list, step_size: float, lags: int, max_inv: int, inventry_aversion: float):
    parser.add_argument("-sz", "--step_size", default=step_size, help="Step size in seconds.", type=float)
    parser.add_argument("-t", "--ticker", default=ticker, help="Specify stock ticker.", type=str)
    parser.add_argument("-ic", "--initial_cash", default=0, help="Initial portfolio.", type=float)
    parser.add_argument("-ii", "--initial_inventory", default=0, help="Initial inventory.", type=float)
    parser.add_argument("-ig", "--initial_gain", default=0, help="Initial gain.", type=float)
    parser.add_argument("-mi", "--max_inventory", default=max_inv, help="Maximum (absolute) inventory.", type=int)
    parser.add_argument("-ia", "--inventory_aversion", default=inventry_aversion, help="Inventory aversion.", type=float)
    parser.add_argument("-n", "--normalisation_on", default=True, help="Normalise features.", type=bool)

    parser.add_argument(
        "-f",
        "--features",
        default="full_state",
        choices=["agent_state", "market_state", "full_state"],
        help="Agent state, market state or full state.",
        type=str,
    )
    parser.add_argument("-nlf", "--n_lags_feature", default=lags, help="Number of lags per feature", type=int)

    parser.add_argument("-starttrain", "--start_trading_train", default=dates[0], help="Start trading train.", type=datetime)
    parser.add_argument("-endtrain", "--end_trading_train", default=dates[1], help="End trading train.", type=datetime)
    parser.add_argument("-endeval", "--start_trading_eval", default=dates[2], help="Start trading eval.", type=datetime)
    parser.add_argument("-starteval", "--end_trading_eval", default=dates[3], help="End trading eval.", type=datetime)


    parser.add_argument(
        "-psr",
        "--per_step_reward_function",
        default="AD",
        choices=["PnL", "AD", "SD"],
        help="Per step reward function: pnl, symm dampened (SD), asymm dampened (AD)",
        type=str,
    )

    parser.add_argument(
        "-mofi",
        "--market_order_fraction_of_inventory",
        default=0.99,
        help="Market order fraction of inventory.",
        type=float,
    )


def get_env_configs(args):
    env_config = {
        "ticker": args["ticker"],
        "start_trading": args["start_trading_train"],
        "end_trading": args["end_trading_train"],
        "step_size": args["step_size"],
        "features": args["features"],
        "max_inventory": args["max_inventory"],
        "inventory_aversion": args["inventory_aversion"],
        "normalisation_on": args["normalisation_on"],
        "initial_cash": args["initial_cash"],
        "initial_inventory": args["initial_inventory"],
        "initial_gain": args["initial_gain"],
        "per_step_reward_function": args["per_step_reward_function"],
        "market_order_fraction_of_inventory": args["market_order_fraction_of_inventory"],
        "n_lags_feature": args["n_lags_feature"]
    }

    eval_env_config = deepcopy(env_config)
    eval_env_config["start_trading"] = args["start_trading_eval"]
    eval_env_config["end_trading"] = args["end_trading_eval"]

    return env_config, eval_env_config


