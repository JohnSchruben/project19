
using Microsoft.VisualStudio.TestTools.UnitTesting;

public static class TurnAngleValidator
{
    // Angles are in degrees
    public const int MAX_TURN_ANGLE = 180;

    public const int MIN_TURN_ANGLE = -180;
    // public const int MAX_TURN_ANGLE = 90; // This will cause the unit test to fail.

    public static bool ValidateTurnAngle(double angle)
    {
        return angle is >= MIN_TURN_ANGLE and <= MAX_TURN_ANGLE;
    }
}

[TestClass]
public class TestTurnAngleValidation
{
    // Boundary case: Max angle of 180
    [TestMethod]
    public void TestBoundaryMaxAngle()
    {
        Assert.IsTrue(TurnAngleValidator.ValidateTurnAngle(180));
    }

    // Boundary case: Min angle of -180
    [TestMethod]
    public void TestBoundaryMinAngle()
    {
        Assert.IsTrue(TurnAngleValidator.ValidateTurnAngle(-180));
    }

    // Edge case: Angle > 180
    [TestMethod]
    public void TestEdgeHighAngle()
    {
        Assert.IsFalse(TurnAngleValidator.ValidateTurnAngle(181));
    }

    // Edge case: Angle < -180
    [TestMethod]
    public void TestEdgeLowAngle()
    {
        Assert.IsFalse(TurnAngleValidator.ValidateTurnAngle(-181));
    }

    // Test an invalid angle
    [TestMethod]
    public void TestInvalidAngle()
    {
        Assert.IsFalse(TurnAngleValidator.ValidateTurnAngle(200));
    }

    // Test a valid angle
    [TestMethod]
    public void TestValidAngle()
    {
        Assert.IsTrue(TurnAngleValidator.ValidateTurnAngle(100));
    }
}