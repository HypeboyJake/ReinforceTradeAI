import gym
import numpy as np


class CoinTradingEnv(gym.Env):
    def __init__(self, data):
        super(CoinTradingEnv, self).__init__()
        self.data = data
        # 0 : Buy, 1 : Sell, 3 : Hold
        self.action_space = gym.spaces.Discrete(3)
        self.initial_cash = 1000  # 초기 자본금
        self.current_cash = self.initial_cash  # 현재 현금
        self.trade_fee = 0.0014  # 거래 수수료
        self.current_coin = 0.0  # 현재 보유 코인 수
        self.min_order_amount = 10  # 최소 주문 금액 ($10)
        self.pyramiding_level = 0 # 피라미딩 횟수
        self.portfolio_value = 0  # 포트폴리오 가치

    def reset(self):
        self.current_step = 0
        self.current_cash = self.initial_cash
        self.current_coin = 0.0
        self.portfolio_valu= 0.0
        self.roe_per_step = []
        return self.data[self.current_step]

    def step(self, action):
        current_price = self.data[self.current_step][3]
        portfolio_value_before = self.current_cash + (self.current_coin * current_price) 
        self.current_cash, self.current_coin, final_action = self.execute_trade(action)
        self.portfolio_value = self.current_cash + (self.current_coin * current_price)
        self.record_roe()

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self.data[self.current_step] if not done else np.zeros_like(self.data[0])

        reward = self.calculate_reward(final_action, portfolio_value_before, self.portfolio_value)##
        return obs, reward, done, {'final_action': final_action}

    def execute_trade(self, action):
        pyramiding_level = self.pyramiding_level
        commission_rate = self.trade_fee
        investment_ratio = 1.0 if pyramiding_level == 0 else 1 / (pyramiding_level + 1)
        final_action = action

        if action == 0: 
            amount_to_invest = self.current_cash * investment_ratio
            amount_to_invest_after_commission = amount_to_invest * (1 - commission_rate)
            if amount_to_invest_after_commission >= self.min_order_amount:
                num_coins_to_buy = amount_to_invest_after_commission / self.data[self.current_step][3]
                if num_coins_to_buy > 0:
                    self.current_cash -= num_coins_to_buy * self.data[self.current_step][3] * (1 + commission_rate)
                    self.current_coin += num_coins_to_buy
                else:
                    final_action = 2 
            else:
                final_action = 2  

        elif action == 1: 
            num_coins_to_sell = self.current_coin * investment_ratio
            if num_coins_to_sell > 0:
                self.current_cash += num_coins_to_sell * self.data[self.current_step][3] * (1 - commission_rate)
                self.current_coin -= num_coins_to_sell
            else:
                final_action = 2  

        self.current_cash = max(self.current_cash, 0)  
        self.current_coin = max(self.current_coin, 0) 

        return self.current_cash, self.current_coin, final_action 



    def calculate_reward(self, final_action, portfolio_value_before, portfolio_value_after):
        long_term_lookback = 20  
        if self.current_step > long_term_lookback:
            long_term_profit_loss = portfolio_value_after / self.data[self.current_step - long_term_lookback][3] - 1
        else:
            long_term_profit_loss = portfolio_value_after / self.initial_cash - 1

        unrealized_reward_weight = 0.5
        realized_reward_weight = 1
        long_term_reward_weight = 2 
        trading_fee = self.trade_fee
        risk_adjustment_factor = 0.05  

        reward = 0
        if final_action == 0: 
            realized_profit_loss = portfolio_value_after - portfolio_value_before
            reward = realized_profit_loss * realized_reward_weight - trading_fee
        elif final_action == 1: 
            realized_profit_loss = portfolio_value_before - portfolio_value_after
            reward = realized_profit_loss * realized_reward_weight - trading_fee
        elif final_action == 2: 
            unrealized_profit_loss = portfolio_value_after - portfolio_value_before
            reward = unrealized_profit_loss * unrealized_reward_weight

        reward += long_term_profit_loss * long_term_reward_weight

        lookback = min(self.current_step, 20)
        past_prices = self.data[self.current_step-lookback:self.current_step]
        past_returns = [(past_prices[i+1][3] - past_prices[i][3]) / past_prices[i][3] for i in range(lookback - 1)]
        volatility = np.std(past_returns) if len(past_returns) > 1 else 0

        risk_adjusted_reward = reward * (1 - risk_adjustment_factor * volatility)

        return risk_adjusted_reward

        
    def record_roe(self):
        total_assets = self.current_cash + (self.current_coin * self.data[self.current_step][3])
        roe = ((total_assets - self.initial_cash) / self.initial_cash) * 100
        self.roe_per_step.append(roe)
        return roe