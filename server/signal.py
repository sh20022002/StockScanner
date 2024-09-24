import scraping

class Signal:
    """
    Represents a trading signal.
    
    Attributes:
        symbol (str): The stock symbol.
        signal_type (str): The type of signal ('buy' or 'sell').
        signal_time (datetime): The time the signal was generated.
    """
    def __init__(self, symbol, signal_type):
        self.symbol = symbol
        self.signal_type = signal_type  # 'buy' or 'sell'
        self.signal_time = scraping.get_exchange_time()

    def __str__(self):
        return f"Signal({self.symbol}, {self.signal_type}, {self.signal_time})"

class SignalStack:
    """
    Represents a stack of trading signals.
    
    Attributes:
        signals (list): A list of Signal objects.
    """
    def __init__(self):
        self.signals = []

    def push(self, signal):
        """
        Pushes a new signal onto the stack and removes irrelevant signals.
        
        Args:
            signal (Signal): The signal to be added.
        """
        self.signals.append(signal)
        self.remove_irrelevant_signals()

    def pop(self):
        """
        Pops the most recent signal from the stack.
        
        Returns:
            Signal: The most recent signal, or None if the stack is empty.
        """
        if not self.is_empty():
            return self.signals.pop()
        return None

    def peek(self):
        """
        Peeks at the most recent signal without removing it.
        
        Returns:
            Signal: The most recent signal, or None if the stack is empty.
        """
        if not self.is_empty():
            return self.signals[-1]
        return None

    def is_empty(self):
        """
        Checks if the stack is empty.
        
        Returns:
            bool: True if the stack is empty, False otherwise.
        """
        return len(self.signals) == 0

    def remove_irrelevant_signals(self):
        """
        Removes signals that are not relevant anymore.
        For example, you could define irrelevance by a time threshold or
        when a new signal of the opposite type is issued for the same symbol.
        """
        # Remove signals older than a certain threshold (e.g., 1 hour)
        current_time = scraping.get_exchange_time()
        self.signals = [signal for signal in self.signals if current_time - signal.signal_time <= timedelta(hours=1)]

        # Alternatively, remove all but the latest signal for each symbol
        latest_signals = {}
        for signal in self.signals:
            latest_signals[signal.symbol] = signal
        self.signals = list(latest_signals.values())

    def __str__(self):
        return "\n".join(str(signal) for signal in self.signals)
