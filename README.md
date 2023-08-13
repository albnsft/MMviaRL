# Market Making via Reinforcement Learning

Market making is a fundamental trading problem in which
an agent continually quotes two-sided markets for securities, providing
bids and asks that he is willing to trade. Market makers provide liquidity
and depth to markets and earn their quoted spread if both orders are executed.
They are compensated for the risk of holding securities, which may
decline in value between the purchase and the sale. The accumulation of
these unfavourable positions is known as inventory risk. In this paper,
we develop a high-fidelity limit order book simulator based on historical
data with 5 levels on each side, and use it to design a market making
agent using temporal-difference learning. We use a Deep Q-learning as a
value function approximator and a custom reward function that
controls inventory risk.

It is set up to use data provided by [LOBSTER](https://lobsterdata.com/).
