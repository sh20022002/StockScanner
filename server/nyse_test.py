import unittest
from datetime import datetime
from scraping import is_nyse_open

class TestIsNYSEOpen(unittest.TestCase):
    def run(self, result=None):
        # Override the run method to raise an exception on failure
        if result is None:
            result = self.defaultTestResult()
        self._feedErrorsToResult(result, self._outcome.errors)
        if result.failures or result.errors:
            raise Exception(f"Test failed: {self}")
        return super().run(result)

    def test_weekday_within_trading_hours(self):
        # Test when it's a weekday and within trading hours
        current_time = datetime(2022, 1, 3, 10, 0)  # Monday, 10:00 AM
        expected_result = True
        self.assertEqual(is_nyse_open(current_time), expected_result)

    def test_weekday_outside_trading_hours(self):
        # Test when it's a weekday but outside trading hours
        current_time = datetime(2022, 1, 3, 8, 0)  # Monday, 8:00 AM
        expected_result = False
        self.assertEqual(is_nyse_open(current_time), expected_result)

    def test_weekend(self):
        # Test when it's a weekend
        current_time = datetime(2022, 1, 1, 10, 0)  # Saturday, 10:00 AM
        expected_result = False
        self.assertEqual(is_nyse_open(current_time), expected_result)

    def test_holiday(self):
        # Test when it's a holiday
        current_time = datetime(2022, 1, 1, 10, 0)  # New Year's Day, 10:00 AM
        expected_result = False
        self.assertEqual(is_nyse_open(current_time), expected_result)

if __name__ == '__main__':
    # Run the tests and capture the results
    runner = unittest.TextTestRunner()
    result = runner.run(unittest.makeSuite(TestIsNYSEOpen))
    
    # Print the results
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures or result.errors:
        raise Exception("Some tests failed.")