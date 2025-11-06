"""
Run all tests
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test suites
try:
    from tests.test_reward import TestRewardFunction
    from tests.test_models import TestModels
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Please ensure all test files and src files exist.")
    sys.exit(1)

if __name__ == "__main__":
    print("="*60)
    print("Running All Tests")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRewardFunction))
    suite.addTest(unittest.makeSuite(TestModels))
    
    # Create runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = runner.run(suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("ALL TESTS PASSED! ✓")
    else:
        print("TESTS FAILED ❌")
        sys.exit(1)
