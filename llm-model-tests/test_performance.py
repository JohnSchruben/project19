"""
Performance tests for real-time LLM inference.
Tests FAIL if models don't meet 20 FPS (≤50ms mean latency) requirement.

Run with: python test_performance.py
Or with pytest: pytest test_performance.py -v
"""

import unittest
import sys
import os
from benchmark import run_benchmark, MODELS
import ollama_utils

# Test configuration
TEST_IMAGE = "car-on-road-3.jpg"
TEST_ITERATIONS = 50 # Increased iterations for more stable latency measurements
MAX_MEAN_LATENCY_MS = 50.0 # 20 FPS = 50ms max
MAX_P95_LATENCY_MS = 70.0 # Allow some outliers but should still be under 70ms for 20 FPS

class TestRealTimePerformance(unittest.TestCase):
    """
    Test cases for real-time performance constraints.
    """
    
    @classmethod
    def setUpClass(cls):
        """Verify test environment before running tests."""
        if not os.path.exists(TEST_IMAGE):
            raise FileNotFoundError(f"Test image not found: {TEST_IMAGE}")
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE TESTS - REAL-TIME CONSTRAINT VALIDATION")
        print(f"{'='*60}")
        print(f"Requirement: 20 FPS (mean latency ≤ {MAX_MEAN_LATENCY_MS}ms)")
        print(f"P95 tolerance: ≤ {MAX_P95_LATENCY_MS}ms")
        print(f"Test iterations: {TEST_ITERATIONS} per model")
        print(f"{'='*60}\n")
    
    def _test_model_performance(self, model_name: str):
        """
        Generic test method for any model.
        Tests FAIL if:
        - Mean latency > 50ms
        - P95 latency > 70ms
        """
        with ollama_utils.OllamaService():
            result = run_benchmark(model_name, TEST_IMAGE, TEST_ITERATIONS)
        
        mean_latency = result["total_mean_ms"]
        p95_latency = result["total_p95_ms"]
        fps = result["fps"]
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Mean Latency: {mean_latency:.2f}ms")
        print(f"  P95 Latency:  {p95_latency:.2f}ms")
        print(f"  FPS: {fps:.2f}")
        
        # Assertions
        self.assertLessEqual(
            mean_latency,
            MAX_MEAN_LATENCY_MS,
            f"{model_name} FAILED: Mean latency {mean_latency:.2f}ms exceeds {MAX_MEAN_LATENCY_MS}ms (required for 20 FPS)"
        )
        
        self.assertLessEqual(
            p95_latency,
            MAX_P95_LATENCY_MS,
            f"{model_name} FAILED: P95 latency {p95_latency:.2f}ms exceeds {MAX_P95_LATENCY_MS}ms tolerance"
        )
        
        print(f"✓ PASS: Meets real-time constraints")

# Dynamically generate test methods for each model
def _create_test_method(model_name):
    """Factory function to create test methods for each model."""
    def test_method(self):
        self._test_model_performance(model_name)
    return test_method

# Add test methods for each model in MODELS list
for model in MODELS:
    # Create valid test method name (replace special chars)
    test_name = f"test_{model.replace('.', '_').replace('-', '_')}_performance"
    test_method = _create_test_method(model)
    test_method.__name__ = test_name
    test_method.__doc__ = f"Test {model} meets 20 FPS real-time constraint"
    setattr(TestRealTimePerformance, test_name, test_method)

class TestPerformanceBreakdown(unittest.TestCase):
    """
    Additional tests to verify performance breakdown.
    """
    
    def test_preprocessing_overhead(self):
        """Verify preprocessing time is negligible (< 5ms)."""
        with ollama_utils.OllamaService():
            result = run_benchmark(MODELS[0], TEST_IMAGE, 20)
        
        preproc_mean = result["preprocessing_mean_ms"]
        self.assertLess(
            preproc_mean,
            5.0,
            f"Preprocessing overhead too high: {preproc_mean:.2f}ms"
        )
    
    def test_postprocessing_overhead(self):
        """Verify postprocessing time is negligible (< 2ms)."""
        with ollama_utils.OllamaService():
            result = run_benchmark(MODELS[0], TEST_IMAGE, 20)
        
        postproc_mean = result["postprocessing_mean_ms"]
        self.assertLess(
            postproc_mean,
            2.0,
            f"Postprocessing overhead too high: {postproc_mean:.2f}ms"
        )
    
    def test_inference_dominates_latency(self):
        """Verify inference time is the main bottleneck (>90% of total)."""
        with ollama_utils.OllamaService():
            result = run_benchmark(MODELS[0], TEST_IMAGE, 20)
        
        inference_pct = (result["inference_mean_ms"] / result["total_mean_ms"]) * 100
        self.assertGreater(
            inference_pct,
            90.0,
            f"Inference should dominate latency but only accounts for {inference_pct:.1f}%"
        )

def main():
    """Run tests with detailed output."""
    # Run with high verbosity
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()