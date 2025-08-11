# 🚀 NewsSense Setup & Execution Guide

## 📋 Prerequisites


### Local Environment
- Python 3.8 or higher
- pip package manager

## ⚡ Quick Start (3 Methods)



### Local Python Environment

1. **Create Project Directory**
   ```bash
   mkdir newssense-system
   cd newssense-system
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   # Windows
   python -m venv newssense_env
   newssense_env\Scripts\activate
   
   # macOS/Linux
   python -m venv newssense_env
   source newssense_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install pydantic
   ```

4. **Create Main File**
   - Save the NewsSense code as `newssense.py`
   
5. **Run the System**
   ```bash
   python newssense.py
   ```

### Method 3: Jupyter Notebook (Local)

1. **Install Jupyter**
   ```bash
   pip install jupyter pydantic
   ```

2. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

3. **Create New Notebook**
   - Copy NewsSense code into cells
   - Run cells sequentially

## 🎮 Interactive Usage Examples

### Basic Interaction
```python
# Initialize the system
from newssense import NewsSenseSystem

system = NewsSenseSystem()

# Ask for trending news
result = system.process_query("What's trending in tech today?")
print(f"Intent: {result['intent']}")
print(f"Agent: {result['routed_agent']}")

# Fact-check a claim
result = system.process_query("Is it true that Apple acquired OpenAI?")
print(f"Verdict: {result['response_data']['verdict']}")

# Summarize content
result = system.process_query("Summarize the latest AI developments")
for point in result['response_data']['summary_points']:
    print(point)
```

### Advanced Usage
```python
# With user tracking
result = system.process_query(
    "Show me finance news", 
    user_id="user123", 
    session_id="session456"
)

# View system logs
logs = system.get_system_logs()
print(f"Total operations logged: {len(logs)}")

# Check processing performance  
print(f"Query processed in: {result['processing_time_ms']:.1f}ms")
```

## 🔧 Customization Options

### Adding New News Sources
```python
# Extend MOCK_NEWS_DATABASE in the code
MOCK_NEWS_DATABASE.append(
    NewsHeadline(
        title="Your Custom News Title",
        source="Your Source",
        url="https://example.com/news",
        category=NewsCategory.TECH,
        relevance_score=0.9
    )
)
```

### Adding Fact-Check Data
```python
# Extend MOCK_FACT_CHECK_DATABASE
MOCK_FACT_CHECK_DATABASE["your claim key"] = {
    "verdict": "TRUE",
    "confidence": 0.95,
    "summary": "Your fact-check summary",
    "supporting": [...],
    "contradicting": [...]
}
```

### Modifying Agent Behavior
```python
# Customize intent patterns
self.intent_patterns[UserIntent.GET_TRENDING].append(r'new pattern')

# Adjust confidence thresholds
if confidence > 0.9:  # High confidence threshold
    # Handle high-confidence queries differently
```

## 📊 Output Examples

### Trending News Response
```json
{
  "intent": "get_trending",
  "routed_agent": "TrendingNewsAgent",
  "response_data": {
    "headlines": [
      {
        "title": "Meta Releases New AI Model Surpassing GPT-4",
        "source": "TechCrunch",
        "url": "https://techcrunch.com/meta-ai-model",
        "category": "tech",
        "relevance_score": 0.95
      }
    ],
    "total_found": 5,
    "query_topic": "ai"
  },
  "processing_time_ms": 45.2,
  "confidence": 0.92
}
```

### Fact-Check Response
```json
{
  "intent": "verify_claim",
  "routed_agent": "FactCheckerAgent", 
  "response_data": {
    "claim": "Apple acquired OpenAI",
    "verdict": "PARTIALLY_TRUE",
    "confidence": 0.75,
    "summary": "While no official partnership has been announced...",
    "supporting_sources": [...],
    "contradicting_sources": [...]
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: pydantic"**
   ```bash
   pip install pydantic
   ```

2. **"Import Error" in Colab**
   - Restart runtime: Runtime → Restart Runtime
   - Re-run installation cell

3. **"Mock data not found"**
   - Ensure you've run the full code including data definitions
   - Check that MOCK_NEWS_DATABASE is populated

### Performance Issues
```python
# Check system logs for bottlenecks
logs = system.get_system_logs()
error_logs = [log for log in logs if log['level'] == 'ERROR']
print(f"Found {len(error_logs)} errors")
```

## 🔄 Development Workflow

### 1. Testing New Features
```python
# Test individual agents
trending_agent = TrendingNewsAgent()
request = TrendingNewsRequest(topic="AI", limit=5)
result = trending_agent.get_trending_news(request)
```

### 2. Adding Real API Integration
```python
# Replace mock tools with real APIs
class RealWebSearchTool:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def search_trending_news(self, request):
        # Implement real web search API call
        pass
```

### 3. Enhanced Logging
```python
# Replace MockLogfire with real Logfire
import logfire

# Configure real logging
logfire.configure(
    service_name="newssense",
    service_version="1.0.0"
)
```

## 📈 Performance Benchmarks

Expected performance on typical hardware:
- **Query Processing**: 20-100ms per query
- **Memory Usage**: ~50MB base + data
- **Concurrent Queries**: Handles 10+ simultaneous requests

### Running Performance Tests
```python
# Built-in performance test
perf_results = run_performance_test()

# Custom performance test
import time
start = time.time()
for i in range(100):
    system.process_query(f"Test query {i}")
print(f"100 queries in {time.time() - start:.2f}s")
```

## 🚀 Next Steps

1. **Try the Demo**: Run `run_demo()` to see all features
2. **Experiment with Queries**: Test different question types
3. **Customize Data**: Add your own news sources and fact-check data
4. **Integrate APIs**: Replace mock tools with real web search and fact-check APIs
5. **Deploy**: Consider deployment options (Flask app, FastAPI, etc.)

## 💡 Sample Queries to Try

```python
# Trending news queries
"What's happening in tech today?"
"Show me trending finance news"
"Latest developments in AI"

# Fact-checking queries  
"Is it true that Meta released a new AI model?"
"Did Apple really acquire OpenAI?"
"Verify: Bitcoin reached $70,000"

# Summarization queries
"Summarize the latest tech news"
"Give me a brief overview of recent developments"
"Key points about current events"

# General queries
"What can you help me with?"
"How does this system work?"
```

Ready to explore the future of news intelligence? Start with the demo and experiment with different queries! 🎯
