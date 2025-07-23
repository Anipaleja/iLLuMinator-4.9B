"""
Performance Monitor for iLLuMinator AI
Analyzes and optimizes model performance for faster inference
"""

import time
import psutil
import gc
from typing import Dict, List, Any
import json
from pathlib import Path

class PerformanceMonitor:
    """Monitor and optimize model performance"""
    
    def __init__(self):
        self.metrics = []
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        gc.collect()  # Clean up memory
        
    def record_inference(self, input_text: str, output_text: str, response_time: float):
        """Record inference metrics"""
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        metric = {
            "timestamp": time.time(),
            "input_length": len(input_text),
            "output_length": len(output_text),
            "response_time": response_time,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage,
            "tokens_per_second": len(output_text.split()) / max(response_time, 0.001)
        }
        
        self.metrics.append(metric)
        
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze collected performance metrics"""
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        response_times = [m["response_time"] for m in self.metrics]
        token_rates = [m["tokens_per_second"] for m in self.metrics]
        memory_usage = [m["memory_usage"] for m in self.metrics]
        
        analysis = {
            "total_inferences": len(self.metrics),
            "average_response_time": sum(response_times) / len(response_times),
            "max_response_time": max(response_times),
            "min_response_time": min(response_times),
            "average_tokens_per_second": sum(token_rates) / len(token_rates),
            "average_memory_usage": sum(memory_usage) / len(memory_usage),
            "max_memory_usage": max(memory_usage),
            "recommendations": self._get_recommendations()
        }
        
        return analysis
    
    def _get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if not self.metrics:
            return ["No data available for recommendations"]
        
        avg_response_time = sum(m["response_time"] for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m["memory_usage"] for m in self.metrics) / len(self.metrics)
        
        # Response time recommendations
        if avg_response_time > 5.0:
            recommendations.append("Response time is slow (>5s). Consider using fast_mode=True or reducing max_tokens")
        elif avg_response_time > 2.0:
            recommendations.append("Response time could be improved. Try reducing model size or using GPU acceleration")
        
        # Memory recommendations
        if avg_memory > 80:
            recommendations.append("High memory usage detected. Consider clearing conversation history more frequently")
        elif avg_memory > 60:
            recommendations.append("Memory usage is moderate. Monitor for memory leaks")
        
        # Token rate recommendations
        avg_token_rate = sum(m["tokens_per_second"] for m in self.metrics) / len(self.metrics)
        if avg_token_rate < 5:
            recommendations.append("Low token generation rate. Check if model is running on GPU and using fast_mode")
        
        if not recommendations:
            recommendations.append("Performance looks good! No specific optimizations needed")
        
        return recommendations
    
    def save_metrics(self, filename: str = "performance_metrics.json"):
        """Save metrics to file"""
        analysis = self.analyze_performance()
        
        data = {
            "analysis": analysis,
            "raw_metrics": self.metrics,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Performance metrics saved to {filename}")

def benchmark_model():
    """Run a quick benchmark of the model"""
    print("=== iLLuMinator AI Performance Benchmark ===")
    
    try:
        from illuminator_ai import IlluminatorAI
        
        # Initialize monitor
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Load model
        print("Loading iLLuMinator AI...")
        load_start = time.time()
        ai = IlluminatorAI(fast_mode=True)
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Test cases for benchmarking
        test_cases = [
            "Hi",
            "What is Python?",
            "Write a simple function",
            "Explain machine learning",
            "How do I optimize code?"
        ]
        
        print("\\nRunning benchmark tests...")
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: '{test_input}'")
            
            start_time = time.time()
            response = ai.chat(test_input)
            response_time = time.time() - start_time
            
            monitor.record_inference(test_input, response, response_time)
            
            print(f"Response time: {response_time:.2f}s")
            print(f"Response length: {len(response)} characters")
            print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            print("-" * 50)
        
        # Analyze results
        print("\\n=== Performance Analysis ===")
        analysis = monitor.analyze_performance()
        
        print(f"Total inferences: {analysis['total_inferences']}")
        print(f"Average response time: {analysis['average_response_time']:.2f}s")
        print(f"Max response time: {analysis['max_response_time']:.2f}s")
        print(f"Min response time: {analysis['min_response_time']:.2f}s")
        print(f"Average tokens/second: {analysis['average_tokens_per_second']:.1f}")
        print(f"Average memory usage: {analysis['average_memory_usage']:.1f}%")
        
        print("\\n=== Recommendations ===")
        for rec in analysis['recommendations']:
            print(f"• {rec}")
        
        # Save metrics
        monitor.save_metrics()
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

def monitor_api_performance():
    """Monitor API server performance"""
    print("=== API Performance Monitor ===")
    print("This tool helps monitor the API server performance")
    print("Run this while the API server is running to collect metrics")
    
    try:
        import requests
        
        api_url = "http://localhost:8000"
        test_requests = [
            {"endpoint": "/health", "method": "GET"},
            {"endpoint": "/chat", "method": "POST", "data": {"message": "Hi"}},
            {"endpoint": "/chat", "method": "POST", "data": {"message": "What is programming?"}},
        ]
        
        monitor = PerformanceMonitor()
        
        for test in test_requests:
            try:
                start_time = time.time()
                
                if test["method"] == "GET":
                    response = requests.get(f"{api_url}{test['endpoint']}")
                else:
                    response = requests.post(f"{api_url}{test['endpoint']}", json=test.get("data", {}))
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    print(f"✓ {test['endpoint']}: {response_time:.2f}s")
                    
                    if "data" in test:
                        input_text = test["data"].get("message", "")
                        output_text = response.json().get("response", "")
                        monitor.record_inference(input_text, output_text, response_time)
                else:
                    print(f"✗ {test['endpoint']}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"✗ {test['endpoint']}: {e}")
        
        # Show analysis
        analysis = monitor.analyze_performance()
        if "error" not in analysis:
            print("\\n=== API Performance Analysis ===")
            print(f"Average response time: {analysis['average_response_time']:.2f}s")
            for rec in analysis['recommendations']:
                print(f"• {rec}")
    
    except ImportError:
        print("requests library not available. Install with: pip install requests")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        monitor_api_performance()
    else:
        benchmark_model()
