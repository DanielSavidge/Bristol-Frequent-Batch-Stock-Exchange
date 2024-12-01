"""Module containing all trader algos"""
# pylint: disable=too-many-lines
import math
import random
import sys
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from tbse_msg_classes import Order
from tbse_sys_consts import TBSE_SYS_MAX_PRICE, TBSE_SYS_MIN_PRICE


# pylint: disable=too-many-instance-attributes
class Trader:
    """Trader superclass - mostly unchanged from original BSE code by Dave Cliff
    all Traders have a trader id, bank balance, blotter, and list of orders to execute"""

    def __init__(self, ttype, tid, balance, time):
        self.ttype = ttype  # what type / strategy this trader is
        self.tid = tid  # trader unique ID code
        self.balance = balance  # money in the bank
        self.blotter = []  # record of trades executed
        self.orders = {}  # customer orders currently being worked (fixed at 1)
        self.n_quotes = 0  # number of quotes live on LOB
        self.willing = 1  # used in ZIP etc
        self.able = 1  # used in ZIP etc
        self.birth_time = time  # used when calculating age of a trader/strategy
        self.profit_per_time = 0  # profit per unit t
        self.n_trades = 0  # how many trades has this trader done?
        self.last_quote = None  # record of what its last quote was
        self.times = [0, 0, 0, 0]  # values used to calculate timing elements

    def log(self, message):
        self.logger.debug(f"{self.ttype}_{self.tid} at time {self.time}: {message}")

    def __str__(self):
        return f'[TID {self.tid} type {self.ttype} balance {self.balance} blotter {self.blotter} ' \
               f'orders {self.orders} n_trades {self.n_trades} profit_per_time {self.profit_per_time}]'

    def add_order(self, order, verbose):
        """
        Adds an order to the traders list of orders
        in this version, trader has at most one order,
        if allow more than one, this needs to be self.orders.append(order)
        :param order: the order to be added
        :param verbose: should verbose logging be printed to console
        :return: Response: "Proceed" if no current offer on LOB, "LOB_Cancel" if there is an order on the LOB needing
                 cancelled.\
        """

        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders[order.coid] = order

        if verbose:
            print(f'add_order < response={response}')
        return response

    def del_order(self, coid):
        """
        Removes current order from traders list of orders
        :param coid: Customer order ID of order to be deleted
        """
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        # CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
        self.orders.pop(coid)

    def bookkeep(self, trade, order, verbose, time):
        """
        Updates trader's internal stats with trade and order
        :param trade: Trade that has been executed
        :param order: Order trade was in response to
        :param verbose: Should verbose logging be printed to console
        :param time: Current time
        """
        output_string = ""

        if trade['coid'] in self.orders:
            coid = trade['coid']
            order_price = self.orders[coid].price
        elif trade['counter'] in self.orders:
            coid = trade['counter']
            order_price = self.orders[coid].price
        else:
            print("COID not found")
            sys.exit("This is non ideal ngl.")

        self.blotter.append(trade)  # add trade record to trader's blotter
        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transaction_price = trade['price']
        if self.orders[coid].otype == 'Bid':
            profit = order_price - transaction_price
        else:
            profit = transaction_price - order_price
        self.balance += profit
        self.n_trades += 1
        self.profit_per_time = self.balance / (time - self.birth_time)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            print(str(trade['coid']) + " " + str(trade['counter']) + " " + str(order.coid) + " " + str(
                self.orders[0].coid))
            sys.exit()

        if verbose:
            print(f'{output_string} profit={profit} balance={self.balance} profit/t={self.profit_per_time}')
        self.del_order(coid)  # delete the order

    # pylint: disable=unused-argument,no-self-use
    def respond(self, time, p_eq, q_eq, demand_curve, supply_curve, lob, trades, verbose):
        """
        specify how trader responds to events in the market
        this is a null action, expect it to be overloaded by specific algos
        :param time: Current time
        :param lob: Limit order book
        :param trade: Trade being responded to
        :param verbose: Should verbose logging be printed to console
        :return: Unused
        """
        return None

    # pylint: disable=unused-argument,no-self-use

    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        """
        Get's the traders order based on the current state of the market
        :param time: Current time
        :param countdown: Time to end of session
        :param lob: Limit order book
        :return: The order
        """
        return None


class TraderZIC(Trader):
    """ Trader subclass ZI-C
        After Gode & Sunder 1993"""

    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        """
        Gets ZIC trader, limit price is randomly selected
        :param time: Current time
        :param countdown: Time until end of current market session
        :param lob: Limit order book
        :return: The trader order to be sent to the exchange
        """

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:

            coid = max(self.orders.keys())

            min_price_lob = lob['bids']['worst']
            max_price_lob = lob['asks']['worst']
            limit = self.orders[coid].price
            otype = self.orders[coid].otype

            min_price = min_price_lob
            max_price = max_price_lob

            if otype == 'Bid':
                if min_price > limit:
                    min_price = min_price_lob
                quote_price = random.randint(min_price, limit)
            else:
                if max_price < limit:
                    max_price = max_price_lob
                quote_price = random.randint(limit, max_price)
                # NB should check it == 'Ask' and barf if not
            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order

        return order


# Trader subclass ZIP
# After Cliff 1997 （adjusted the beta and momentum）
# pylint: disable=too-many-instance-attributes
class TraderZip(Trader):
    """ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
    NB this implementation keeps separate margin values for buying & selling,
       so a single trader can both buy AND sell
       -- in the original, traders were either buyers OR sellers"""

    def __init__(self, ttype, tid, balance, time):

        Trader.__init__(self, ttype, tid, balance, time)
        m_fix = 0.05
        m_var = 0.3
        self.job = None  # this is 'Bid' or 'Ask' depending on customer order
        self.active = False  # gets switched to True while actively working an order
        self.prev_change = 0  # this was called last_d in Cliff'97
        self.beta = 0.1 + 0.2 * random.random()  # Original 0.2 + 0.2 * random.random()
        self.momentum = 0.2 * random.random()  # Original 0.3 * random.random()
        self.ca = 0.10  # self.ca & .cr were hard-coded in '97 but parameterised later
        self.cr = 0.10
        self.margin = None  # this was called profit in Cliff'97
        self.margin_buy = -1.0 * (m_fix + m_var * random.random())
        self.margin_sell = m_fix + m_var * random.random()
        self.price = None
        self.limit = None
        self.times = [0, 0, 0, 0]
        # memory of best price & quantity of best bid and ask, on LOB on previous update
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None
        self.last_batch = None

    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        """
        :param time: Current time
        :param countdown: Time until end of current market session
        :param lob: Limit order book
        :return: Trader order to be sent to exchange
        """
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            coid = max(self.orders.keys())
            self.active = True
            self.limit = self.orders[coid].price
            self.job = self.orders[coid].otype
            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                self.margin = self.margin_buy
            else:
                # currently a seller (working a sell order)
                self.margin = self.margin_sell
            quote_price = int(self.limit * (1 + self.margin))
            self.price = quote_price

            order = Order(self.tid, self.job, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order
        return order

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def respond(self, time, p_eq, q_eq, demand_curve, supply_curve, lob, trades, verbose):
        """
        update margin on basis of what happened in marke
        ZIP trader responds to market events, altering its margin
        does this whether it currently has an order to work or not
        :param time: Current time
        :param lob: Limit order book
        :param trade: Trade being responded to
        :param verbose: Should verbose logging be printed to console
        """

        if self.last_batch == (demand_curve, supply_curve):
            return
        else:
            self.last_batch = (demand_curve, supply_curve)

        trade = trades[0] if trades else None

        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if demand_curve != []:
            best_bid = max(demand_curve, key=lambda x: x[0])[0]
        if supply_curve != []:
            best_ask = min(supply_curve, key=lambda x: x[0])[0]

        def target_up(price):
            """
            generate a higher target price by randomly perturbing given price
            :param price: Current price
            :return: New price target
            """
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel + ptrb_abs, 0))

            return target

        def target_down(price):
            """
            generate a lower target price by randomly perturbing given price
            :param price: Current price
            :return: New price target
            """
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel - ptrb_abs, 0))

            return target

        def willing_to_trade(price):
            """
            am I willing to trade at this price?
            :param price: Price to be traded out
            :return: Is the trader willing to trade
            """
            willing = False
            if self.job == 'Bid' and self.active and self.price >= price:
                willing = True
            if self.job == 'Ask' and self.active and self.price <= price:
                willing = True
            return willing

        def profit_alter(price):
            """
            Update target profit margin
            :param price: New target profit margin
            """
            old_price = self.price
            diff = price - old_price
            change = ((1.0 - self.momentum) * (self.beta * diff)) + (self.momentum * self.prev_change)
            self.prev_change = change
            new_margin = ((self.price + change) / self.limit) - 1.0

            if self.job == 'Bid':
                if new_margin < 0.0:
                    self.margin_buy = new_margin
                    self.margin = new_margin
            else:
                if new_margin > 0.0:
                    self.margin_sell = new_margin
                    self.margin = new_margin

            # set the price from limit and profit-margin
            self.price = int(round(self.limit * (1.0 + self.margin), 0))

        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False

        # lob_best_bid_p = lob['bids']['best']
        lob_best_bid_p = best_bid  # CHANGE HERE
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = 1
            if self.prev_best_bid_p is None:
                self.prev_best_bid_p = lob_best_bid_p
            elif self.prev_best_bid_p < lob_best_bid_p:
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]  # might have to check if has been cancelled at some point during batch
            # for item in lob['tape'] check if cancel happened with price of
            if last_tape_item['type'] == 'Cancel':
                # print("Last bid was cancelled") #test.csv
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        # lob_best_ask_p = lob['asks']['best']
        lob_best_ask_p = best_ask  # CHANGE HERE
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = 1
            if self.prev_best_ask_p is None:
                self.prev_best_ask_p = lob_best_ask_p
            elif self.prev_best_ask_p > lob_best_ask_p:
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                # print("Last bid was cancelled") # test.csv
                ask_lifted = False
            else:
                ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted

        if trade is None:
            deal = False

        if self.job == 'Ask':
            # seller
            if deal:
                trade_price = trade['price']
                if self.price <= trade_price:
                    # could sell for more? raise margin
                    target_price = target_up(trade_price)
                    profit_alter(target_price)
                elif ask_lifted and self.active and not willing_to_trade(trade_price):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_down(trade_price)
                    profit_alter(target_price)
            else:
                # no deal: aim for a target price higher than best bid
                if ask_improved and self.price > lob_best_ask_p:
                    if lob_best_bid_p is not None:
                        target_price = target_up(lob_best_bid_p)
                    else:
                        target_price = lob['asks']['worst']  # stub quote
                    profit_alter(target_price)

        if self.job == 'Bid':
            # buyer
            if deal:
                trade_price = trade['price']
                if self.price >= trade_price:
                    # could buy for less? raise margin (i.e. cut the price)
                    target_price = target_down(trade_price)
                    profit_alter(target_price)
                elif bid_hit and self.active and not willing_to_trade(trade_price):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_up(trade_price)
                    profit_alter(target_price)
            else:
                # no deal: aim for target price lower than best ask
                if bid_improved and self.price < lob_best_bid_p:
                    if lob_best_ask_p is not None:
                        target_price = target_down(lob_best_ask_p)
                    else:
                        target_price = lob['bids']['worst']  # stub quote
                    profit_alter(target_price)

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

# Predict the equilibrium price
# pylint: disable=too-many-instance-attributes
class TraderRaForest(Trader):
    def __init__(self, ttype, tid, balance, time):
        super().__init__(ttype, tid, balance, time)
        self.model = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=42) # original version 10, 5
        self.history = []
        self.min_history = 15
        self.max_history = 50
        self.last_batch = None
        self.training_frequency = 15

    # Limit the amount of historical data and retrain after every training_frequency data collection to speed up the efficiency
    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        if len(self.orders) < 1:
            self.active = False
            return None

        self.active = True
        coid = max(self.orders.keys())
        limit_price = self.orders[coid].price
        otype = self.orders[coid].otype

        market_data = self.collect_market_data(time, p_eq, q_eq, demand_curve, supply_curve, lob)
        self.history.append(market_data)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        if len(self.history) < self.min_history:
            return Order(self.tid, otype, limit_price, self.orders[coid].qty, time, coid, self.orders[coid].toid)

        if len(self.history) % self.training_frequency == 0:
            X = np.array(self.history[:-1])
            y = np.array([data[0] for data in self.history[1:]])
            self.model.fit(X, y)

        try:
            prediction = self.model.predict(np.array(market_data).reshape(1, -1))[0]
            if otype == 'Bid': # Ensure that the price limit is not exceeded
                quote_price = min(int(prediction), limit_price)
            else:  # otype == 'Ask'
                quote_price = max(int(prediction), limit_price)
        except Exception as e:
            print(f"Prediction error: {e}. Using limit price.")
            quote_price = limit_price

        order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, coid, self.orders[coid].toid)
        self.last_quote = order
        return order

    def respond(self, time, p_eq, q_eq, demand_curve, supply_curve, lob, trades, verbose):
        pass

    # Collect the 4 key data as the feature
    def collect_market_data(self, time, p_eq, q_eq, demand_curve, supply_curve, lob):
        best_bid = max(demand_curve, key=lambda x: x[0])[0] if demand_curve else p_eq
        best_ask = min(supply_curve, key=lambda x: x[0])[0] if supply_curve else p_eq
        return [p_eq, q_eq, best_bid, best_ask]


# GVWY is always powerful in Savidge’s results. I designed this algorithm in the hope that GVWY would be more adaptable
class TraderMIX(Trader):
    def __init__(self, ttype, tid, balance, time):
        super().__init__(ttype, tid, balance, time)
        self.strategies = {
            'GVWY': self.gvwy_strategy,
            'IFB': self.ifb_strategy,
            'Adaptive': self.adaptive_strategy
        }
        self.market_state_history = []
        self.performance_history = {strategy: [] for strategy in self.strategies}
        self.price_history = []
        self.learning_rate = 0.1
        self.order_book_imbalance_threshold = 0.2
        self.volatility_threshold = 0.02
        self.vwap_window = 10
        self.previous_balance = balance
        self.last_strategy = None


    # Check the market state
    def classify_market_state(self, order_imbalance, volatility):
        if abs(order_imbalance) < 0.1 and volatility < self.volatility_threshold / 2:
            return 'Stable'
        elif abs(order_imbalance) > 0.3 or volatility > self.volatility_threshold:
            return 'Volatile'
        else:
            return 'Normal'

    # choose the strategy which in this state and based on the history performance
    def decide_strategy(self, market_state):
        state_performance = {}
        for strategy in self.strategies:
            performances = [p for s, p in zip(self.market_state_history, self.performance_history[strategy])
                            if s == market_state]
            if performances:
                state_performance[strategy] = np.mean(performances[-10:])
            else:
                state_performance[strategy] = 0  # Default performance if no data
        return max(state_performance, key=state_performance.get)

    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        if len(self.orders) < 1:
            return None

        coid = max(self.orders.keys())
        limit_price = self.orders[coid].price
        otype = self.orders[coid].otype

        order_imbalance = self.analyze_order_book(demand_curve, supply_curve)
        volatility = self.calculate_volatility(p_eq)
        market_state = self.classify_market_state(order_imbalance, volatility)
        self.market_state_history.append(market_state)

        chosen_strategy = self.decide_strategy(market_state)
        self.last_strategy = chosen_strategy

        if chosen_strategy == 'GVWY':
            quote_price = self.gvwy_strategy(limit_price)
        elif chosen_strategy == 'IFB':
            quote_price = self.ifb_strategy(limit_price, otype, order_imbalance)
        else:
            quote_price = self.adaptive_strategy(p_eq, limit_price, otype, order_imbalance, volatility)

        order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, coid, self.orders[coid].toid)
        self.last_quote = order
        return order

    def analyze_order_book(self, demand_curve, supply_curve):
        if not demand_curve or not supply_curve:
            return 0
        bid_volume = sum(q for _, q in demand_curve)
        ask_volume = sum(q for _, q in supply_curve)
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0

    # calculate the market volatility
    # VWAP window is 10, so elder data will be deleted
    def calculate_volatility(self, p_eq):
        self.price_history.append(p_eq)
        if len(self.price_history) > self.vwap_window:
            self.price_history = self.price_history[-self.vwap_window:]
        if len(self.price_history) > 1:
            return np.std(self.price_history) / np.mean(self.price_history)
        return 0

    def gvwy_strategy(self, limit_price):
        return limit_price

    def ifb_strategy(self, limit_price, otype, order_imbalance):
        if abs(order_imbalance) > self.order_book_imbalance_threshold:
            adjustment_factor = 1 + (0.1 * abs(order_imbalance))  # Dynamic factor of preference
            if order_imbalance > 0:  # Bullish signal
                if otype == 'Bid':
                    return limit_price
                else:
                    return int(limit_price * adjustment_factor)
            else:  # Bearish signal
                if otype == 'Bid':
                    return int(limit_price / adjustment_factor)
                else:
                    return limit_price
        else:
            return limit_price

    # Adaptive strategy
    # When the market is unstable (volatile), we use VWAP conservative, because the VWAP gives average price during the period
    # By adjusting the alpha parameter, can change the algorithm's preference for historical and current prices
    def adaptive_strategy(self, p_eq, limit_price, otype, order_imbalance, volatility):
        vwap = np.mean(self.price_history) if self.price_history else p_eq
        alpha = 0.5  # Adaptive parameter
        target_price = alpha * vwap + (1 - alpha) * p_eq

        adjustment = 1 + (order_imbalance * volatility * (-1 if otype == 'Bid' else 1))
        quote_price = int(target_price * adjustment)

        if otype == 'Bid':
            return min(quote_price, limit_price)
        else:
            return max(quote_price, limit_price)

    def bookkeep(self, trade, order, verbose, time):
        super().bookkeep(trade, order, verbose, time)
        profit = self.balance - self.previous_balance
        self.performance_history[self.last_strategy].append(profit)
        self.previous_balance = self.balance

    def respond(self, time, p_eq, q_eq, demand_curve, supply_curve, lob, trades, verbose):
        pass


class TraderGiveaway(Trader):
    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        """
        Get's giveaway traders order - in this case the price is just the limit price from the customer order
        :param time: Current time
        :param countdown: Time until end of session
        :param lob: Limit order book
        :return: Order to be sent to the exchange
        """

        if len(self.orders) < 1:
            order = None
        else:
            coid = max(self.orders.keys())  # Find the price limit
            quote_price = self.orders[coid].price
            order = Order(self.tid,
                          self.orders[coid].otype,
                          quote_price,
                          self.orders[coid].qty,
                          time, self.orders[coid].coid, self.orders[coid].toid)
            self.last_quote = order
        return order


class TraderAA(Trader):
    """
        Daniel Snashall's implementation of Vytelingum's AA trader, first described in his 2006 PhD Thesis.
        For more details see: Vytelingum, P., 2006. The Structure and Behaviour of the Continuous Double
        Auction. PhD Thesis, University of Southampton
        """
    # Parameters Changed for the sensitivity analysis
    def __init__(self, ttype, tid, balance, time):
        # Stuff about trader
        super().__init__(ttype, tid, balance, time)
        self.active = False

        self.limit = None
        self.job = None

        # learning variables
        self.r_shout_change_relative = 0.05
        self.r_shout_change_absolute = 0.05
        self.short_term_learning_rate = random.uniform(0.5, 0.9)  # 0.3 0.7
        self.long_term_learning_rate = random.uniform(0.01, 0.1)  # 0.1 0.5
        self.moving_average_weight_decay = 0.7  # how fast weight decays with t, lower is quicker, 0.9 in vytelingum
        self.moving_average_window_size = 5
        self.offer_change_rate = 3.0
        self.theta = -2.0
        self.theta_max = 3.0  # 2.0
        self.theta_min = -10.0  # -8.0
        self.market_max = TBSE_SYS_MAX_PRICE

        # Variables to describe the market
        self.previous_transactions = []
        self.moving_average_weights = []
        for i in range(self.moving_average_window_size):
            self.moving_average_weights.append(self.moving_average_weight_decay ** i)
        self.estimated_equilibrium = []
        self.smiths_alpha = []
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

        # Trading Variables
        self.r_shout = None
        self.buy_target = None
        self.sell_target = None
        self.buy_r = -1.0 * (0.3 * random.random())
        self.sell_r = -1.0 * (0.3 * random.random())

        # define last batch so that internal values are only updated upon new batch matching
        self.last_batch = None

    def calc_eq(self):
        """
        Calculates the estimated 'eq' or estimated equilibrium price.
        Slightly modified from paper, it is unclear in paper
        N previous transactions * weights / N in Vytelingum, swap N denominator for sum of weights to be correct?
        :return: Estimated equilibrium price
        """
        if len(self.previous_transactions) == 0:
            return
        if len(self.previous_transactions) < self.moving_average_window_size:
            # Not enough transactions
            self.estimated_equilibrium.append(
                float(sum(self.previous_transactions)) / max(len(self.previous_transactions), 1))
        else:
            n_previous_transactions = self.previous_transactions[-self.moving_average_window_size:]
            thing = [n_previous_transactions[i] * self.moving_average_weights[i] for i in
                     range(self.moving_average_window_size)]
            eq = sum(thing) / sum(self.moving_average_weights)
            self.estimated_equilibrium.append(eq)

    def calc_alpha(self):
        """
        Calculates trader's alpha value - see AA paper for details.
        """
        alpha = 0.0
        for p in self.estimated_equilibrium:
            alpha += (p - self.estimated_equilibrium[-1]) ** 2
        alpha = math.sqrt(alpha / len(self.estimated_equilibrium))
        self.smiths_alpha.append(alpha / self.estimated_equilibrium[-1])

    def calc_theta(self):
        """
        Calculates trader's theta value - see AA paper for details.
        """
        gamma = 2.0  # not sensitive apparently so choose to be whatever
        # necessary for initialisation, div by 0
        if min(self.smiths_alpha) == max(self.smiths_alpha):
            alpha_range = 0.4  # starting value i guess
        else:
            alpha_range = (self.smiths_alpha[-1] - min(self.smiths_alpha)) / (
                    max(self.smiths_alpha) - min(self.smiths_alpha))
        theta_range = self.theta_max - self.theta_min
        desired_theta = self.theta_min + theta_range * (1 - (alpha_range * math.exp(gamma * (alpha_range - 1))))
        self.theta = self.theta + self.long_term_learning_rate * (desired_theta - self.theta)

    def calc_r_shout(self):
        """
        Calculates trader's r shout value - see AA paper for details.
        """
        p = self.estimated_equilibrium[-1]
        lim = self.limit
        theta = self.theta
        if self.job == 'Bid':
            # Currently a buyer
            if lim <= p:  # extra-marginal!
                self.r_shout = 0.0
            else:  # intra-marginal :(
                if self.buy_target > self.estimated_equilibrium[-1]:
                    # r[0,1]
                    self.r_shout = math.log(((self.buy_target - p) * (math.exp(theta) - 1) / (lim - p)) + 1) / theta
                else:
                    # r[-1,0]
                    self.r_shout = math.log((1 - (self.buy_target / p)) * (math.exp(theta) - 1) + 1) / theta

        if self.job == 'Ask':
            # Currently a seller
            if lim >= p:  # extra-marginal!
                self.r_shout = 0
            else:  # intra-marginal :(
                if self.sell_target > self.estimated_equilibrium[-1]:
                    # r[-1,0]
                    self.r_shout = math.log(
                        (self.sell_target - p) * (math.exp(theta) - 1) / (self.market_max - p) + 1) / theta
                else:
                    # r[0,1]
                    a = (self.sell_target - lim) / (p - lim)
                    self.r_shout = (math.log((1 - a) * (math.exp(theta) - 1) + 1)) / theta

    def calc_agg(self):
        """
        Calculates Trader's aggressiveness parameter - see AA paper for details.
        """
        if self.job == 'Bid':
            # BUYER
            if self.buy_target >= self.previous_transactions[-1]:
                # must be more aggressive
                delta = (1 + self.r_shout_change_relative) * self.r_shout + self.r_shout_change_absolute
            else:
                delta = (1 - self.r_shout_change_relative) * self.r_shout - self.r_shout_change_absolute

            self.buy_r = self.buy_r + self.short_term_learning_rate * (delta - self.buy_r)

        if self.job == 'Ask':
            # SELLER
            if self.sell_target > self.previous_transactions[-1]:
                delta = (1 + self.r_shout_change_relative) * self.r_shout + self.r_shout_change_absolute
            else:
                delta = (1 - self.r_shout_change_relative) * self.r_shout - self.r_shout_change_absolute

            self.sell_r = self.sell_r + self.short_term_learning_rate * (delta - self.sell_r)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def calc_target(self):
        """
        Calculates trader's target price - see AA paper for details.
        """
        p = 1
        if len(self.estimated_equilibrium) > 0:
            p = self.estimated_equilibrium[-1]
            if self.limit == p:
                p = p * 1.000001  # to prevent theta_bar = 0
        elif self.job == 'Bid':
            p = self.limit - self.limit * 0.2  # Initial guess for eq if no deals yet!!....
        elif self.job == 'Ask':
            p = self.limit + self.limit * 0.2
        lim = self.limit
        theta = self.theta
        if self.job == 'Bid':
            # BUYER
            minus_thing = (math.exp(-self.buy_r * theta) - 1) / (math.exp(theta) - 1)
            plus_thing = (math.exp(self.buy_r * theta) - 1) / (math.exp(theta) - 1)
            theta_bar = (theta * lim - theta * p) / p
            if theta_bar == 0:
                theta_bar = 0.0001
            if math.exp(theta_bar) - 1 == 0:
                theta_bar = 0.0001
            bar_thing = (math.exp(-self.buy_r * theta_bar) - 1) / (math.exp(theta_bar) - 1)
            if lim <= p:  # Extra-marginal
                if self.buy_r >= 0:
                    self.buy_target = lim
                else:
                    self.buy_target = lim * (1 - minus_thing)
            else:  # intra-marginal
                if self.buy_r >= 0:
                    self.buy_target = p + (lim - p) * plus_thing
                else:
                    self.buy_target = p * (1 - bar_thing)
            self.buy_target = min(self.buy_target, lim)

        if self.job == 'Ask':
            # SELLER
            minus_thing = (math.exp(-self.sell_r * theta) - 1) / (math.exp(theta) - 1)
            plus_thing = (math.exp(self.sell_r * theta) - 1) / (math.exp(theta) - 1)
            theta_bar = (theta * lim - theta * p) / p
            if theta_bar == 0:
                theta_bar = 0.0001
            if math.exp(theta_bar) - 1 == 0:
                theta_bar = 0.0001
            bar_thing = (math.exp(-self.sell_r * theta_bar) - 1) / (math.exp(theta_bar) - 1)  # div 0 sometimes what!?
            if lim <= p:  # Extra-marginal
                if self.buy_r >= 0:
                    self.buy_target = lim
                else:
                    self.buy_target = lim + (self.market_max - lim) * minus_thing
            else:  # intra-marginal
                if self.buy_r >= 0:
                    self.buy_target = lim + (p - lim) * (1 - plus_thing)
                else:
                    self.buy_target = p + (self.market_max - p) * bar_thing
            if self.sell_target is None:
                self.sell_target = lim
            elif self.sell_target < lim:
                self.sell_target = lim

    # pylint: disable=too-many-branches
    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        """
        Creates an AA trader's order
        :param time: Current time
        :param countdown: Time left in the current trading period
        :param lob: Current state of the limit order book
        :return: Order to be sent to the exchange
        """
        if len(self.orders) < 1:
            self.active = False
            return None
        coid = max(self.orders.keys())
        self.active = True
        self.limit = self.orders[coid].price
        self.job = self.orders[coid].otype
        self.calc_target()

        if self.prev_best_bid_p is None:
            o_bid = 0
        else:
            o_bid = self.prev_best_bid_p
        if self.prev_best_ask_p is None:
            o_ask = self.market_max
        else:
            o_ask = self.prev_best_ask_p

        quote_price = TBSE_SYS_MIN_PRICE
        if self.job == 'Bid':  # BUYER
            if self.limit <= o_bid:
                return None
            if len(self.previous_transactions) > 0:  # has been at least one transaction
                o_ask_plus = (1 + self.r_shout_change_relative) * o_ask + self.r_shout_change_absolute
                quote_price = o_bid + ((min(self.limit, o_ask_plus) - o_bid) / self.offer_change_rate)
            else:
                if o_ask <= self.buy_target:
                    quote_price = o_ask
                else:
                    quote_price = o_bid + ((self.buy_target - o_bid) / self.offer_change_rate)
        elif self.job == 'Ask':
            if self.limit >= o_ask:
                return None
            if len(self.previous_transactions) > 0:  # has been at least one transaction
                o_bid_minus = (1 - self.r_shout_change_relative) * o_bid - self.r_shout_change_absolute
                quote_price = o_ask - ((o_ask - max(self.limit, o_bid_minus)) / self.offer_change_rate)
            else:
                if o_bid >= self.sell_target:
                    quote_price = o_bid
                else:
                    quote_price = o_ask - ((o_ask - self.sell_target) / self.offer_change_rate)

        order = Order(self.tid, self.job, int(quote_price), self.orders[coid].qty, time, self.orders[coid].coid,
                      self.orders[coid].toid)
        self.last_quote = order
        return order

    # pylint: disable=too-many-branches
    def respond(self, time, p_eq, q_eq, demand_curve, supply_curve, lob, trades, verbose):
        """
        Updates AA trader's internal variables based on activities on the LOB
        Beginning nicked from ZIP
        what, if anything, has happened on the bid LOB? Nicked from ZIP.
        :param time: current time
        :param lob: current state of the limit order book
        :param trade: trade which occurred to trigger this response
        :param verbose: should verbose logging be printed to the console
        """

        if self.last_batch == (demand_curve, supply_curve):
            return
        else:
            self.last_batch = (demand_curve, supply_curve)

        trade = trades[0] if trades else None

        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if demand_curve != []:
            best_bid = max(demand_curve, key=lambda x: x[0])[0]
        if supply_curve != []:
            best_ask = min(supply_curve, key=lambda x: x[0])[0]

        bid_hit = False
        # lob_best_bid_p = lob['bids']['best'] #CHANGED
        lob_best_bid_p = best_bid
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = 1
            if self.prev_best_bid_p is None:
                self.prev_best_bid_p = lob_best_bid_p
            # elif self.prev_best_bid_p < lob_best_bid_p :
            #     # best bid has improved
            #     # NB doesn't check if the improvement was by self
            #     bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        # ask_improved = False
        ask_lifted = False

        # lob_best_ask_p = lob['asks']['best'] #CHANGED THIS
        lob_best_ask_p = best_ask
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = 1
            if self.prev_best_ask_p is None:
                self.prev_best_ask_p = lob_best_ask_p
            # elif self.prev_best_ask_p > lob_best_ask_p :
            #     # best ask has improved -- NB doesn't check if the improvement was by self
            #     ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                ask_lifted = False
            else:
                ask_lifted = True

        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

        deal = bid_hit or ask_lifted
        if (trades == []):
            deal = False

        # End nicked from ZIP

        if deal:
            # if trade is not None:
            self.previous_transactions.append(trade['price'])
            if self.sell_target is None:
                self.sell_target = trade['price']  # CHANGED THIS
                # self.sell_target = best_ask
            if self.buy_target is None:
                self.buy_target = trade['price']  # CHANGED THIS
                # self.sell_target = best_bid
            self.calc_eq()
            self.calc_alpha()
            self.calc_theta()
            self.calc_r_shout()
            self.calc_agg()
            self.calc_target()

    # ----------------trader-types have all been defined now-------------


class TraderRandom(Trader):
    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        if len(self.orders) < 1:
            order = None
        else:
            coid = max(self.orders.keys())
            limit = self.orders[coid].price
            otype = self.orders[coid].otype
            if otype == 'Bid':
                quote_price = random.randint(TBSE_SYS_MIN_PRICE, limit)  # Totally random
            else:  # otype == 'Ask'
                quote_price = random.randint(limit, TBSE_SYS_MAX_PRICE)

            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order

        return order


class TraderHerd(Trader):
    def __init__(self, ttype, tid, balance, time):
        super().__init__(ttype, tid, balance, time)

    def get_order(self, time, p_eq, q_eq, demand_curve, supply_curve, countdown, lob):
        if len(self.orders) < 1:
            return None

        coid = max(self.orders.keys())
        limit = self.orders[coid].price
        otype = self.orders[coid].otype


        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if best_bid is None or best_ask is None:

            quote_price = limit
        else:

            if otype == 'Bid':

                quote_price = min(best_bid + 1, best_ask - 1, limit)
            else:  # otype == 'Ask'

                quote_price = max(best_ask - 1, best_bid + 1, limit)

        order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                      self.orders[coid].toid)
        self.last_quote = order

        return order


