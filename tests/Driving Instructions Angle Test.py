#Tests the validation of the one of the outputs of the LLM, the driving turn angle.
import unittest

# Angles are in degrees
MAX_TURN_ANGLE = 180  # This will cause the unit test to pass.
MIN_TURN_ANGLE = -180
# MAX_TURN_ANGLE = 90  # This will cause the unit test to fail.

'''
Turn Angle validation function
angle: int or float of the turn angle in degrees.
'''
def validate_turn_angle(angle):
    # Check if angle is a number that between MIN_TURN_ANGLE and MAX_TURN_ANGLE (inclusive)
    return isinstance(angle, (float, int)) and MIN_TURN_ANGLE <= angle <= MAX_TURN_ANGLE

# Unit Test Class
class TestTurnAngleValidation(unittest.TestCase):

    # Test a valid angle
    def test_valid_angle(self):
        self.assertTrue(validate_turn_angle(100))

    # Test an invalid angle
    def test_invalid_angle(self):
        self.assertFalse(validate_turn_angle(200))

    # Boundary case: Max angle of 180
    def test_boundary_max_angle(self):
        self.assertTrue(validate_turn_angle(180))
        
    # Boundary case: Min angle of -180
    def test_boundary_min_angle(self):
        self.assertTrue(validate_turn_angle(-180))

    # Edge case: Percentage > 180
    def test_edge_high_angle(self):
        self.assertFalse(validate_turn_angle(181))
        
    # Edge case: Percentage < -180
    def test_edge_low_angle(self):
        self.assertFalse(validate_turn_angle(-181))

if __name__ == "__main__":
    print("\nRUNNING UNIT TESTS")
    unittest.main() # Run the tests