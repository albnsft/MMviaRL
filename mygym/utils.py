from datetime import timedelta

from database.HistoricalDatabase import HistoricalDatabase
from simulation.OrderbookSimulator import OrderbookSimulator
from mygym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from rewards.RewardFunctions import InventoryAdjustedPnL, PnL

from features.Features import Portfolio

from pylab import plt
import pandas as pd
import numpy as np
import os


def get_reward_function(reward_function: str, inventory_aversion: float = 0.1):
    if reward_function == "AD":  # asymmetrically dampened
        return InventoryAdjustedPnL(inventory_aversion=inventory_aversion, asymmetrically_dampened=True)
    elif reward_function == "SD":  # symmetrically dampened
        return InventoryAdjustedPnL(inventory_aversion=inventory_aversion, asymmetrically_dampened=False)
    elif reward_function == "PnL":
        return PnL()
    else:
        raise NotImplementedError("You must specify one of 'AS', 'SD', 'PnL'")


def env_creator(env_config):
    database = HistoricalDatabase(ticker=env_config["ticker"])

    if env_config["features"] == "agent_state":
        features = HistoricalOrderbookEnvironment.get_default_features(
            step_size=timedelta(seconds=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )[-3:]

    elif env_config["features"] == "market_state":
        features = HistoricalOrderbookEnvironment.get_default_features(
            step_size=timedelta(seconds=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )[:-3]

    elif env_config["features"] == "full_state":
        features = HistoricalOrderbookEnvironment.get_default_features(
            step_size=timedelta(seconds=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )

    orderbook_simulator = OrderbookSimulator(
        ticker=env_config["ticker"],
        database=database,
    )
    env = HistoricalOrderbookEnvironment(
        start_of_trading=env_config["start_trading"],
        end_of_trading=env_config["end_trading"],
        max_inventory=env_config["max_inventory"],
        ticker=env_config["ticker"],
        simulator=orderbook_simulator,
        features=features,
        step_size=timedelta(seconds=env_config["step_size"]),
        market_order_fraction_of_inventory=env_config["market_order_fraction_of_inventory"],
        initial_portfolio=Portfolio(inventory=env_config["initial_inventory"], cash=env_config["initial_cash"],
                                    gain=env_config["initial_gain"]),
        per_step_reward_function=get_reward_function(env_config["per_step_reward_function"],
                                                     env_config["inventory_aversion"]),
        n_lags_feature=env_config["n_lags_feature"]
    )
    return env


def done_inf(dct):
    df = pd.DataFrame.from_dict(dct, orient='index').T
    df.index = list(df.index + 1)
    return np.round(df, 1)


def plot_per_episode(
        ticker,
        agent_name,
        step_size,
        market_order_clearing,
        reward_fun,
        step_info_per_episode,
        step_info_per_eval_episode,
        episode,
        done_info,
        done_info_eval,
):
    step_info_per_episode = step_info_per_episode[episode] if step_info_per_episode is not None else None
    step_info_per_eval_episode = step_info_per_eval_episode[episode]

    def join_df(step_info_per_episode, step_info_per_eval_episode, train_metric, val_metric):
        if step_info_per_episode is not None:
            train_metric = pd.DataFrame([train_metric], index=['training'],
                                        columns=step_info_per_episode.__dict__['dates'][-len(train_metric):]).T
            val_metric = pd.DataFrame([val_metric], index=['testing'],
                                      columns=step_info_per_eval_episode.__dict__['dates'][-len(val_metric):]).T
            metrics = pd.concat([train_metric, val_metric])
            assert (len(metrics.T) == 2)
            assert (len(metrics) == (len(train_metric) + len(val_metric)))
        else:
            train_metric = pd.DataFrame()
            val_metric = pd.DataFrame([val_metric], index=['testing'],
                                      columns=step_info_per_eval_episode.__dict__['dates'][-len(val_metric):]).T
            metrics = val_metric
        return train_metric, val_metric, metrics

    def graph_per_episode(step_info_per_episode, step_info_per_eval_episode, metric: str = None):
        train_metric = step_info_per_episode.__dict__[metric] if step_info_per_episode is not None else None
        val_metric = step_info_per_eval_episode.__dict__[metric]
        _, _, metrics = join_df(step_info_per_episode, step_info_per_eval_episode, train_metric, val_metric)
        return metrics

    def info_metrics(step_info_per_episode, step_info_per_eval_episode, window: str = '10s'):
        train_reward = np.diff(step_info_per_episode.__dict__['pnls']) if step_info_per_episode is not None else None
        val_reward = np.diff(step_info_per_eval_episode.__dict__['pnls'])
        train_reward, test_reward, rewards = join_df(step_info_per_episode, step_info_per_eval_episode, train_reward,
                                                     val_reward)
        train_aum = step_info_per_episode.__dict__['aums'] if step_info_per_episode is not None else None
        val_aum = step_info_per_eval_episode.__dict__['aums']
        train_aum, test_aum, aums = join_df(step_info_per_episode, step_info_per_eval_episode, train_aum, val_aum)
        train_inv = step_info_per_episode.__dict__['inventories'] if step_info_per_episode is not None else None
        val_inv = step_info_per_eval_episode.__dict__['inventories']
        train_inv, test_inv, invs = join_df(step_info_per_episode, step_info_per_eval_episode, train_inv, val_inv)
        map_roll = pd.concat([train_inv.abs().rolling(1).mean(),
                              test_inv.abs().rolling(1).mean()]).dropna(how='all')
        aum_map = (aums / map_roll).dropna(how='all')
        stats = pd.DataFrame(rewards).describe()
        stats = np.round(stats)
        stats = stats.astype(int)
        return stats, rewards, aum_map

    def info_actions(step_info_per_episode, step_info_per_eval_episode):
        actions_train = pd.DataFrame.from_dict(step_info_per_episode.__dict__['actions'],
                                               orient='index').T if step_info_per_episode is not None else pd.DataFrame()
        actions_test = pd.DataFrame.from_dict(step_info_per_eval_episode.__dict__['actions'], orient='index').T
        return actions_train.value_counts(), actions_test.value_counts()

    def info_factions(step_info_per_episode, step_info_per_eval_episode):
        actions_train = pd.DataFrame.from_dict(step_info_per_episode.__dict__['filled_actions'],
                                               orient='index').T if step_info_per_episode is not None else pd.DataFrame()
        factions_train = actions_train[actions_train != -1][actions_train != 0].dropna()
        factions_train_ = actions_train[(actions_train == -1).sum(axis=1).astype('bool')]
        actions_test = pd.DataFrame.from_dict(step_info_per_eval_episode.__dict__['filled_actions'], orient='index').T
        factions_test = actions_test[actions_test != -1][actions_test != 0].dropna()
        factions_test_ = actions_test[(actions_test == -1).sum(axis=1).astype('bool')]
        return factions_train.value_counts(), factions_train_.value_counts(), factions_test.value_counts(), factions_test_.value_counts()

    def uncertainties_aum_map(step_info):
        cols = step_info.__dict__['dates'][1:]
        aum = pd.Series(np.diff(step_info.__dict__['aums']), index=cols)
        inventories = pd.Series(np.diff(step_info.__dict__['inventories']), index=cols)
        map = inventories.abs()

        uncertainties = pd.DataFrame(index=['PnL', 'MAP'],
                                     columns=['Mean', 'Standard Deviation', 'Mean Absolute Deviation']).T
        uncertainties['PnL'] = [np.mean(aum), np.std(aum),
                                np.mean(np.absolute(aum - np.mean(aum)))]
        uncertainties['MAP'] = [np.mean(map), np.std(map),
                                np.mean(np.absolute(map - np.mean(map)))]
        return np.round(uncertainties, 1)

    # sns.set()
    try:
        damp_factor = reward_fun.inventory_aversion
    except:
        damp_factor = None

    if damp_factor:
        reward_fun = 'Asymmetrically dampened PnL' if reward_fun.asymmetrically_dampened else 'Symmetrically dampened PnL'
    else:
        reward_fun = 'PnL'

    if step_info_per_episode is not None:
        fig = plt.figure(constrained_layout=True, figsize=(10, 20))
        ax_dict = fig.subplot_mosaic(
            """
            ZY
            AB
            CD
            EE
            GH
            PQ
            IJ
            """
        )
    else:
        fig = plt.figure(constrained_layout=True, figsize=(10, 15))
        ax_dict = fig.subplot_mosaic(
            """
            ZY
            AB
            CD
            GH
            IJ
            """
        )

    eval_str = 'Testing: ' if done_info is None else ''
    name = f"{eval_str}{ticker} - {agent_name} \n step size: {step_size.total_seconds()}sec | " \
           + f"reward: {reward_fun} | dampening factor: {damp_factor}  | order clearing factor: {market_order_clearing}\n"

    plt.suptitle(name)

    done_info = done_inf(done_info).iloc[episode - 1].to_frame().T if done_info is not None else pd.DataFrame(
        np.zeros(4)).T
    done_info_eval = done_inf(done_info_eval).iloc[episode - 1].to_frame().T
    done_info_eval.drop(['nd_pnl', 'depth'], axis=1, inplace=True)

    if step_info_per_episode is not None:
        table = ax_dict["Z"].table(
            cellText=done_info.values,
            colLabels=done_info.columns,
            loc="center",
        )
        table.set_fontsize(6.5)
        # table.scale(0.5, 1.1)
        ax_dict["Z"].set_axis_off()
        ax_dict["Z"].title.set_text("Agent's characteristics training")

        table = ax_dict["Z"].table(
            cellText=done_info_eval.values,
            colLabels=done_info_eval.columns,
            loc="center",
        )
        table.set_fontsize(6.5)
        # table.scale(0.5, 1.1)
        ax_dict["Z"].set_axis_off()
        ax_dict["Z"].title.set_text("Agent's characteristics testing")

    pnl_curve = graph_per_episode(step_info_per_episode, step_info_per_eval_episode, 'pnls')
    equity_curve = graph_per_episode(step_info_per_episode, step_info_per_eval_episode, 'aums')
    curves = pd.concat([pnl_curve, equity_curve], axis=1)
    if step_info_per_episode is not None:
        curves.columns = ['pnl_training', 'pnl_testing', 'aum_training', 'aum_testing']
        curves.plot(ax=ax_dict["A"], ylabel='$', xlabel='n-th bar reaches',
                    title=f'Cumulative reward & PnL')
    else:
        curves.columns = ['reward', 'pnl']
        curves.plot(ax=ax_dict["B"], ylabel='$',
                    title=f'Cumulative reward & PnL')

    inventory_curve = graph_per_episode(step_info_per_episode, step_info_per_eval_episode, 'inventories')
    if step_info_per_episode is not None:
        inventory_curve.plot(ax=ax_dict["B"], ylabel='inventory',
                             title=f'Inventory curve through time')
    else:
        """
        mid_price = graph_per_episode(step_info_per_episode, step_info_per_eval_episode, 'mid_price')
        mid_inv = pd.concat([mid_price, inventory_curve], axis=1)
        mid_inv.columns = ['mid-price', 'inventory']
        mid_inv.plot(ax=ax_dict["Y"], ylabel='$', secondary_y=['inventory'],
                     title=f'Mid price and agent inventory')
        """

    window = '30min'
    stats_rewards, rewards, aum_map_roll = info_metrics(step_info_per_episode,
                                                        step_info_per_eval_episode,
                                                        window)

    table = ax_dict["C"].table(
        cellText=stats_rewards.values,
        rowLabels=stats_rewards.index,
        colLabels=stats_rewards.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    # table.scale(0.5, 1.1)
    ax_dict["C"].set_axis_off()
    ax_dict["C"].title.set_text("Satistics of agent's reward (PnL)")

    non_null_reward = rewards[rewards != 0].dropna(how='all')
    non_null_reward.plot.hist(ax=ax_dict["D"], title="Non null rewards histogram", bins=50, alpha=0.5)

    letter = 'A' if step_info_per_episode is None else 'E'
    aum_map_roll.iloc[100:].plot(ax=ax_dict[letter], title=f'PnL-to-MAP ratio')

    actions_train, actions_test = info_actions(step_info_per_episode, step_info_per_eval_episode)
    actions_f2train, actions_f1train, actions_f2test, actions_f1test = info_factions(step_info_per_episode,
                                                                                     step_info_per_eval_episode)
    actions_train, actions_test = actions_train.to_frame(
        name="Training"), actions_test.to_frame(name="Testing")
    actions_f2train, actions_f2test = actions_f2train.to_frame(
        name="Training"), actions_f2test.to_frame(name="Testing")
    actions_f1train, actions_f1test = actions_f1train.to_frame(
        name="Training"), actions_f1test.to_frame(name="Testing")
    actions = pd.concat([actions_train, actions_test], axis=1).sort_values(by='Testing')
    actions_2filled = pd.concat([actions_f2train, actions_f2test], axis=1).sort_values(by='Testing')
    actions_1filled = pd.concat([actions_f1train, actions_f1test], axis=1).sort_values(by='Testing')

    actions.plot.barh(ax=ax_dict["G"], title="Count agent's parameter actions (bid, ask)")

    try:
        actions.drop(np.where(actions.index.get_level_values(0).values == 0)[0], level=1,
                     inplace=True)  # delete market orders to count the pct of filled order, indeed market orders are always filled
    except KeyError:
        pass
    np.seterr(divide='ignore', invalid='ignore')
    not_filled = (
                             actions.sum().values - actions_2filled.sum().values - actions_1filled.sum().values) / actions.sum().values
    not_filled = np.round(
        pd.DataFrame(not_filled * 100, index=['Training', 'Testing'], columns=['Not filled actions (%)']).T, 1)
    # not_filled.plot.barh(ax=ax_dict["P"], title = "Not filled actions in units")

    if step_info_per_episode is not None:
        table = ax_dict["P"].table(
            cellText=not_filled.values,
            colLabels=not_filled.columns,
            loc="center",
        )
        table.set_fontsize(6.5)
        # table.scale(0.5, 1.1)
        ax_dict["P"].set_axis_off()
        ax_dict["P"].title.set_text("Agent's actions not filled in %")
    else:
        done_info_eval['actions not filled (%)'] = not_filled['Testing'].values
        table = ax_dict["Z"].table(
            cellText=done_info_eval.values,
            colLabels=done_info_eval.columns,
            loc="center",
        )
        table.set_fontsize(6.5)
        # table.scale(0.5, 1.1)
        ax_dict["Z"].set_axis_off()
        ax_dict["Z"].title.set_text("Agent's characteristics")

    try:
        actions_2filled.plot.barh(ax=ax_dict["H"],
                                  title=f"Count agent's actions filled both side (bid, ask) \n during the step size: {step_size.total_seconds()}sec")
        letter = 'I' if step_info_per_episode is None else 'H'
        actions_1filled.plot.barh(ax=ax_dict[letter],
                                  title=f"Count agent's actions filled one side (bid, ask) \n during the step size: {step_size.total_seconds()}sec")
    except IndexError:
        pass

    if step_info_per_episode is not None:
        uncertainties_train = uncertainties_aum_map(step_info_per_episode)
        table = ax_dict["I"].table(
            cellText=uncertainties_train.values,
            rowLabels=uncertainties_train.index,
            colLabels=uncertainties_train.columns,
            loc="center",
        )
        table.set_fontsize(6.5)
        # table.scale(0.5, 1.1)
        ax_dict["I"].set_axis_off()
        ax_dict["I"].title.set_text(f"Training statistics of AUM and MAP")

    uncertainties_test = uncertainties_aum_map(step_info_per_eval_episode)
    table = ax_dict["J"].table(
        cellText=uncertainties_test.values,
        rowLabels=uncertainties_test.index,
        colLabels=uncertainties_test.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    # table.scale(0.5, 1.1)
    ax_dict["J"].set_axis_off()
    ax_dict["J"].title.set_text("Uncertainties")

    pdf_path = os.path.join("results", agent_name)
    os.makedirs(pdf_path, exist_ok=True)
    subname = name.replace('\n', '').replace(' ', '_').replace(':', '').replace('|', '')
    pdf_filename = os.path.join(pdf_path, f"Ep_{episode}_{subname}.jpg")
    # Write plot to pdf
    fig.savefig(pdf_filename)
    plt.close(fig)


def plot_final(
        done_info_eval,
        ticker,
        agent_name,
        step_size,
        market_order_clearing,
        reward_fun,
):
    def graph_final(done_info_eval, metric):
        metrics = pd.DataFrame([done_info_eval[metric]], index=['testing']).T
        return metrics

    fig = plt.figure(constrained_layout=True, figsize=(15, 15))
    ax_dict = fig.subplot_mosaic(
        """
        AB
        CD
        """
    )

    try:
        damp_factor = reward_fun.inventory_aversion
    except:
        damp_factor = None
    if damp_factor:
        reward_fun = 'Asymmetrically dampened PnL' if reward_fun.asymmetrically_dampened else 'Symmetrically dampened PnL'
    else:
        reward_fun = 'PnL'
    name = f"{ticker} - {agent_name}\n step size in sec: {step_size.total_seconds()} | " \
           + f"reward: {reward_fun} | dampening factor: {damp_factor}  | market order clearing factor: {market_order_clearing}\n"
    plt.suptitle(name)
    pdf_path = os.path.join("results", agent_name)
    subname = name.replace('\n', '').replace(' ', '_').replace(':', '').replace('|', '')
    pdf_filename = os.path.join(pdf_path, f"final_{subname}.pdf")

    metrics = ['nd_pnl', 'depth', 'aum', 'map']
    for metric, ax in zip(metrics, ["A", "B", "C", "D"]):
        graph = graph_final(done_info_eval, metric)
        x = range(1, len(graph) + 1)
        y = np.polyval(np.polyfit(x, graph['testing'], deg=3), x)
        graph.plot(ax=ax_dict[ax], ylabel=metric, xlabel='n-th episodes', title=f'{metric} through episodes')
        ax_dict[ax].plot(x, y, 'r--', label=f'regression {metric}')

    os.makedirs(pdf_path, exist_ok=True)
    fig.savefig(pdf_filename)
    plt.close(fig)


def info_eval(step_info_per_eval_episode, episode, name):
    info = step_info_per_eval_episode[episode]
    pnl = pd.DataFrame(info.aums, index=info.dates, columns=[name])
    inv = pd.DataFrame(info.inventories, index=info.dates, columns=[name])
    return pnl, inv


def plot_eval(pnls_msft,
              invs_msft,
              pnls_goog,
              invs_goog):

    pnls_msft = pd.concat(pnls_msft, axis=1)
    invs_msft = pd.concat(invs_msft, axis=1)
    pnls_goog = pd.concat(pnls_goog, axis=1)
    invs_goog = pd.concat(invs_goog, axis=1)

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    ax_dict = fig.subplot_mosaic(
        """
        AB
        CD
        """
    )

    name = f"Comparison of performance criteria between different agents all evaluated on MSFT and GOOG"
    plt.suptitle(name)
    pdf_path = "results"
    pdf_filename = os.path.join(pdf_path, f"Perf_criteria_testing_set_different_strategies.pdf")

    pnls_msft.plot(ax=ax_dict["A"], ylabel='$', title=f'PnL on MSFT testing set for the considered strategies')
    pnls_goog.plot(ax=ax_dict["C"], ylabel='$', title=f'PnL on GOOG testing set for the considered strategies')
    invs_msft.plot(ax=ax_dict["B"], ylabel='units', title=f'Inventory on MSFT testing set for the considered strategies')
    invs_goog.plot(ax=ax_dict["D"], ylabel='units', title=f'Inventory on GOOG testing set for the considered strategies')

    os.makedirs(pdf_path, exist_ok=True)
    fig.savefig(pdf_filename)
    plt.close(fig)
