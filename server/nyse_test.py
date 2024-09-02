import unittest
from scraping import is_nyse_open  # Ensure this is correctly imported from your module

from unittest.mock import patch
from datetime import datetime, timedelta
from strategy import ExpirableItem, ExpirableStack  # Replace with the actual module name

class TestExpirableItem(unittest.TestCase):

    @patch('strategy.scraping.get_exchange_time')
    def test_is_expired_within_an_hour(self, mock_get_exchange_time):
        # Simulate the current time and item creation time
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 10, 0)  # 10:00 AM
        item = ExpirableItem(value=100, symbol="AAPL", action="buy")
        
        # Simulate the check within one hour
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 10, 30)  # 10:30 AM
        self.assertFalse(item.is_expired())  # Should not be expired

    @patch('strategy.scraping.get_exchange_time')
    def test_is_expired_after_an_hour(self, mock_get_exchange_time):
        # Simulate the current time and item creation time
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 10, 0)  # 10:00 AM
        item = ExpirableItem(value=100, symbol="AAPL", action="buy")
        
        # Simulate the check after one hour
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 11, 1)  # 11:01 AM
        self.assertTrue(item.is_expired())  # Should be expired

class TestExpirableStack(unittest.TestCase):

    @patch('strategy.scraping.get_exchange_time')
    def test_push_and_pop(self, mock_get_exchange_time):
        # Simulate the time
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 10, 0)  # 10:00 AM
        stack = ExpirableStack()

        # Push an item
        stack.push(value=100, symbol="AAPL", action="buy")
        self.assertEqual(stack.size(), 1)

        # Pop the item
        popped_item = stack.pop()
        self.assertEqual(popped_item.value, 100)
        self.assertEqual(popped_item.symbol, "AAPL")
        self.assertEqual(popped_item.action, "buy")
        self.assertTrue(stack.is_empty())

    @patch('strategy.scraping.get_exchange_time')
    def test_remove_expired_items(self, mock_get_exchange_time):
        stack = ExpirableStack()

        # Simulate pushing multiple items at different times
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 9, 0)
        stack.push(value=100, symbol="AAPL", action="buy")

        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 9, 30)
        stack.push(value=200, symbol="GOOGL", action="sell")

        # Simulate time after 1 hour for first item
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 10, 1)
        stack.remove_expired_items()

        # First item should be removed
        self.assertEqual(stack.size(), 1)
        self.assertEqual(stack.peek().symbol, "GOOGL")

    @patch('strategy.scraping.get_exchange_time')
    def test_peek_does_not_remove_unexpired_items(self, mock_get_exchange_time):
        # Simulate the time
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 10, 0)  # 10:00 AM
        stack = ExpirableStack()

        # Push an item
        stack.push(value=100, symbol="AAPL", action="buy")
        
        # Peek should not remove the item if it's not expired
        self.assertEqual(stack.peek().value, 100)
        self.assertEqual(stack.size(), 1)

    @patch('strategy.scraping.get_exchange_time')
    def test_pop_removes_expired_items(self, mock_get_exchange_time):
        stack = ExpirableStack()

        # Simulate pushing an item
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 9, 0)
        stack.push(value=100, symbol="AAPL", action="buy")

        # Simulate time after 1 hour for first item
        mock_get_exchange_time.return_value = datetime(2022, 1, 1, 10, 1)
        popped_item = stack.pop()

        # The stack should now be empty as the only item was expired and should have been removed
        self.assertIsNone(popped_item)
        self.assertTrue(stack.is_empty())

if __name__ == '__main__':
    unittest.main()
