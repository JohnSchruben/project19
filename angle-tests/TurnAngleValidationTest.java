package project19;
public class TurnAngleValidationTest {

    // Angles are in degrees
    static final double MAX_TURN_ANGLE = 180;
    static final double MIN_TURN_ANGLE = -180;

    // Validation function
    public static boolean validateTurnAngle(double angle) {
        return angle >= MIN_TURN_ANGLE && angle <= MAX_TURN_ANGLE;
    }

    // Simple test runner
    public static void main(String[] args) {
        System.out.println("RUNNING UNIT TESTS");

        test("test_valid_angle", validateTurnAngle(100), true);
        test("test_invalid_angle", validateTurnAngle(200), false);
        test("test_boundary_max_angle", validateTurnAngle(180), true);
        test("test_boundary_min_angle", validateTurnAngle(-180), true);
        test("test_edge_high_angle", validateTurnAngle(181), false);
        test("test_edge_low_angle", validateTurnAngle(-181), false);

        System.out.println("ALL TESTS COMPLETED");
    }

    // Helper method
    static void test(String name, boolean actual, boolean expected) {
        if (actual == expected) {
            System.out.println(name + ": PASS");
        } else {
            System.out.println(name + ": FAIL");
        }
    }
}
