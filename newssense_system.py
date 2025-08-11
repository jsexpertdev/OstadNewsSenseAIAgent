# NewsSense Multi-Agent System
# A comprehensive news tracking, verification, and summarization system

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal
from enum import Enum
from dataclasses import dataclass, asdict
import random

# Pydantic for validation
from pydantic import BaseModel, Field, validator

# Mock Logfire implementation (replace with actual logfire in production)
class MockLogfire:
    def __init__(self):
        self.logs = []
    
    def info(self, message: str, **kwargs):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": message,
            "data": kwargs
        }
        self.logs.append(log_entry)
        print(f"[INFO] {message} | {kwargs}")
    
    def error(self, message: str, **kwargs):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR", 
            "message": message,
            "data": kwargs
        }
        self.logs.append(log_entry)
        print(f"[ERROR] {message} | {kwargs}")
    
    def get_logs(self):
        return self.logs

# Initialize mock logfire
logfire = MockLogfire()

# Pydantic Models for Agent Communication
class UserIntent(str, Enum):
    GET_TRENDING = "get_trending"
    VERIFY_CLAIM = "verify_claim" 
    SUMMARIZE_NEWS = "summarize_news"
    GENERAL_QUERY = "general_query"

class NewsCategory(str, Enum):
    TECH = "tech"
    POLITICS = "politics"
    FINANCE = "finance"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    HEALTH = "health"
    GENERAL = "general"

class NewsHeadline(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    source: str = Field(..., min_length=2, max_length=100)
    url: str = Field(..., regex=r'^https?://')
    category: NewsCategory
    timestamp: datetime = Field(default_factory=datetime.now)
    relevance_score: float = Field(..., ge=0.0, le=1.0)

class TrendingNewsRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=100)
    category: Optional[NewsCategory] = NewsCategory.GENERAL
    limit: int = Field(default=10, ge=1, le=50)

class TrendingNewsResponse(BaseModel):
    headlines: List[NewsHeadline]
    total_found: int
    query_topic: str
    generated_at: datetime = Field(default_factory=datetime.now)

class FactCheckRequest(BaseModel):
    claim: str = Field(..., min_length=5, max_length=500)
    context: Optional[str] = None

class FactCheckSource(BaseModel):
    url: str
    title: str
    excerpt: str
    credibility_score: float = Field(..., ge=0.0, le=1.0)

class FactCheckResponse(BaseModel):
    claim: str
    verdict: Literal["TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIED", "INSUFFICIENT_DATA"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_sources: List[FactCheckSource]
    contradicting_sources: List[FactCheckSource]
    summary: str
    checked_at: datetime = Field(default_factory=datetime.now)

class SummarizeRequest(BaseModel):
    article_text: str = Field(..., min_length=100, max_length=10000)
    summary_length: Literal["brief", "detailed"] = "brief"

class SummarizeResponse(BaseModel):
    original_length: int
    summary_points: List[str] = Field(..., min_items=3, max_items=7)
    key_entities: List[str]
    summary_length_type: str
    summarized_at: datetime = Field(default_factory=datetime.now)

class ConversationRequest(BaseModel):
    user_query: str = Field(..., min_length=1, max_length=1000)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ConversationResponse(BaseModel):
    intent: UserIntent
    routed_agent: str
    response_data: Union[TrendingNewsResponse, FactCheckResponse, SummarizeResponse, Dict]
    processing_time_ms: float
    confidence: float = Field(..., ge=0.0, le=1.0)

# Mock Data for Simulation
MOCK_NEWS_DATABASE = [
    NewsHeadline(title="Meta Releases New AI Model Surpassing GPT-4", source="TechCrunch", url="https://techcrunch.com/meta-ai-model", category=NewsCategory.TECH, relevance_score=0.95),
    NewsHeadline(title="Apple Stock Hits All-Time High After iPhone 16 Launch", source="Bloomberg", url="https://bloomberg.com/apple-stock", category=NewsCategory.TECH, relevance_score=0.88),
    NewsHeadline(title="OpenAI and Microsoft Announce Deeper Integration", source="The Verge", url="https://theverge.com/openai-microsoft", category=NewsCategory.TECH, relevance_score=0.92),
    NewsHeadline(title="Bitcoin Surges Past $70,000 Amid ETF Approval", source="CoinDesk", url="https://coindesk.com/bitcoin-surge", category=NewsCategory.FINANCE, relevance_score=0.87),
    NewsHeadline(title="Federal Reserve Hints at Interest Rate Cut", source="Wall Street Journal", url="https://wsj.com/fed-rates", category=NewsCategory.FINANCE, relevance_score=0.85),
    NewsHeadline(title="Climate Summit Reaches Historic Carbon Agreement", source="Reuters", url="https://reuters.com/climate-summit", category=NewsCategory.SCIENCE, relevance_score=0.93),
    NewsHeadline(title="NASA Confirms Water on Mars Subsurface", source="Space.com", url="https://space.com/mars-water", category=NewsCategory.SCIENCE, relevance_score=0.91),
    NewsHeadline(title="Major Healthcare Breakthrough in Gene Therapy", source="Nature", url="https://nature.com/gene-therapy", category=NewsCategory.HEALTH, relevance_score=0.89),
]

MOCK_FACT_CHECK_DATABASE = {
    "apple openai partnership": {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.75,
        "summary": "While no official partnership has been announced, multiple sources confirm ongoing discussions between Apple and OpenAI for potential AI integrations.",
        "supporting": [
            FactCheckSource(url="https://bloomberg.com/apple-openai-talks", title="Apple-OpenAI Talks Progress", excerpt="Sources close to negotiations confirm...", credibility_score=0.9),
        ],
        "contradicting": [
            FactCheckSource(url="https://techcrunch.com/no-apple-deal", title="No Deal Yet Says Apple", excerpt="Apple spokesperson denies current partnership...", credibility_score=0.85),
        ]
    },
    "meta new ai model": {
        "verdict": "TRUE",
        "confidence": 0.92,
        "summary": "Meta has officially announced and released their new AI model, showing improved performance benchmarks.",
        "supporting": [
            FactCheckSource(url="https://meta.com/official-announcement", title="Meta AI Official Release", excerpt="Today we announce our most advanced...", credibility_score=0.95),
        ],
        "contradicting": []
    }
}

# Tool Implementations
class WebSearchTool:
    """Mock web search tool for trending news"""
    
    def search_trending_news(self, request: TrendingNewsRequest) -> TrendingNewsResponse:
        logfire.info("WebSearchTool: Searching trending news", topic=request.topic, category=request.category)
        
        # Filter mock data based on request
        filtered_headlines = []
        for headline in MOCK_NEWS_DATABASE:
            if request.category != NewsCategory.GENERAL and headline.category != request.category:
                continue
            
            # Simple keyword matching
            if request.topic.lower() in headline.title.lower() or request.topic.lower() in headline.category.value:
                filtered_headlines.append(headline)
        
        # Sort by relevance and limit results
        filtered_headlines.sort(key=lambda x: x.relevance_score, reverse=True)
        limited_headlines = filtered_headlines[:request.limit]
        
        response = TrendingNewsResponse(
            headlines=limited_headlines,
            total_found=len(filtered_headlines),
            query_topic=request.topic
        )
        
        logfire.info("WebSearchTool: Found trending news", count=len(limited_headlines), total_available=len(filtered_headlines))
        return response

class RAGTool:
    """Mock RAG tool for fact checking"""
    
    def fact_check_claim(self, request: FactCheckRequest) -> FactCheckResponse:
        logfire.info("RAGTool: Fact checking claim", claim=request.claim)
        
        claim_key = self._extract_key_terms(request.claim)
        
        # Look for matching fact check data
        fact_data = None
        for key, data in MOCK_FACT_CHECK_DATABASE.items():
            if any(term in claim_key for term in key.split()):
                fact_data = data
                break
        
        if not fact_data:
            # Default response for unknown claims
            response = FactCheckResponse(
                claim=request.claim,
                verdict="INSUFFICIENT_DATA",
                confidence=0.3,
                supporting_sources=[],
                contradicting_sources=[],
                summary="Insufficient reliable sources found to verify this claim. More investigation needed."
            )
        else:
            response = FactCheckResponse(
                claim=request.claim,
                verdict=fact_data["verdict"],
                confidence=fact_data["confidence"],
                supporting_sources=fact_data["supporting"],
                contradicting_sources=fact_data["contradicting"],
                summary=fact_data["summary"]
            )
        
        logfire.info("RAGTool: Fact check completed", verdict=response.verdict, confidence=response.confidence)
        return response
    
    def _extract_key_terms(self, claim: str) -> str:
        """Extract key terms from claim for matching"""
        # Simple keyword extraction - remove stop words and convert to lowercase
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "did", "do", "does"}
        words = re.findall(r'\b\w+\b', claim.lower())
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(key_words)

class SummarizationTool:
    """Mock summarization tool"""
    
    def summarize_news(self, request: SummarizeRequest) -> SummarizeResponse:
        logfire.info("SummarizationTool: Summarizing article", length=len(request.article_text), summary_type=request.summary_length)
        
        # Mock summarization - extract sentences and create bullet points
        sentences = self._split_into_sentences(request.article_text)
        entities = self._extract_entities(request.article_text)
        
        if request.summary_length == "brief":
            num_points = min(3, len(sentences))
        else:
            num_points = min(5, len(sentences))
        
        # Select most important sentences (mock importance scoring)
        important_sentences = sentences[:num_points]
        summary_points = [f"• {sentence.strip()}" for sentence in important_sentences]
        
        response = SummarizeResponse(
            original_length=len(request.article_text),
            summary_points=summary_points,
            key_entities=entities[:5],  # Top 5 entities
            summary_length_type=request.summary_length
        )
        
        logfire.info("SummarizationTool: Summarization completed", points_generated=len(summary_points), entities_found=len(entities))
        return response
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Mock entity extraction"""
        # Simple pattern matching for entities (names, companies, etc.)
        entities = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)  # Names
        entities.extend(re.findall(r'\b[A-Z]{2,}\b', text))  # Acronyms
        return list(set(entities))

# Agent Implementations
class TrendingNewsAgent:
    def __init__(self):
        self.web_search_tool = WebSearchTool()
    
    def get_trending_news(self, request: TrendingNewsRequest) -> TrendingNewsResponse:
        logfire.info("TrendingNewsAgent: Processing request", agent="trending_news", topic=request.topic)
        
        try:
            response = self.web_search_tool.search_trending_news(request)
            logfire.info("TrendingNewsAgent: Successfully retrieved trending news", headlines_count=len(response.headlines))
            return response
        except Exception as e:
            logfire.error("TrendingNewsAgent: Error retrieving trending news", error=str(e))
            raise

class FactCheckerAgent:
    def __init__(self):
        self.rag_tool = RAGTool()
    
    def fact_check_claim(self, request: FactCheckRequest) -> FactCheckResponse:
        logfire.info("FactCheckerAgent: Processing fact check", agent="fact_checker", claim=request.claim[:50] + "...")
        
        try:
            response = self.rag_tool.fact_check_claim(request)
            logfire.info("FactCheckerAgent: Fact check completed", verdict=response.verdict, confidence=response.confidence)
            return response
        except Exception as e:
            logfire.error("FactCheckerAgent: Error during fact checking", error=str(e))
            raise

class NewsSummarizerAgent:
    def __init__(self):
        self.summarization_tool = SummarizationTool()
    
    def summarize_news(self, request: SummarizeRequest) -> SummarizeResponse:
        logfire.info("NewsSummarizerAgent: Processing summarization", agent="summarizer", text_length=len(request.article_text))
        
        try:
            response = self.summarization_tool.summarize_news(request)
            logfire.info("NewsSummarizerAgent: Summarization completed", summary_points=len(response.summary_points))
            return response
        except Exception as e:
            logfire.error("NewsSummarizerAgent: Error during summarization", error=str(e))
            raise

class ConversationAgent:
    """Main controller agent that routes user queries to appropriate specialist agents"""
    
    def __init__(self):
        self.trending_agent = TrendingNewsAgent()
        self.fact_checker_agent = FactCheckerAgent()
        self.summarizer_agent = NewsSummarizerAgent()
        
        # Intent classification patterns
        self.intent_patterns = {
            UserIntent.GET_TRENDING: [
                r'\b(trending|popular|latest|breaking|recent|current)\b.*\b(news|headlines|stories)\b',
                r'what\'s happening in',
                r'show me.*news',
                r'trending.*\b(tech|politics|finance|sports|entertainment|science|health)\b'
            ],
            UserIntent.VERIFY_CLAIM: [
                r'\b(is it true|verify|fact check|check if|confirm)\b',
                r'did.*\b(really|actually)\b',
                r'true or false',
                r'verify.*claim'
            ],
            UserIntent.SUMMARIZE_NEWS: [
                r'\b(summarize|summary|brief|outline)\b',
                r'tell me about.*in brief',
                r'key points',
                r'tldr'
            ]
        }
    
    def process_conversation(self, request: ConversationRequest) -> ConversationResponse:
        start_time = datetime.now()
        logfire.info("ConversationAgent: Processing user query", query=request.user_query[:100] + "...", user_id=request.user_id)
        
        try:
            # Classify user intent
            intent, confidence = self._classify_intent(request.user_query)
            logfire.info("ConversationAgent: Intent classified", intent=intent.value, confidence=confidence)
            
            # Route to appropriate agent
            routed_agent, response_data = self._route_to_agent(intent, request.user_query)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = ConversationResponse(
                intent=intent,
                routed_agent=routed_agent,
                response_data=response_data,
                processing_time_ms=processing_time,
                confidence=confidence
            )
            
            logfire.info("ConversationAgent: Query processed successfully", 
                        intent=intent.value, 
                        routed_agent=routed_agent,
                        processing_time_ms=processing_time)
            return response
            
        except Exception as e:
            logfire.error("ConversationAgent: Error processing conversation", error=str(e))
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Return error response
            return ConversationResponse(
                intent=UserIntent.GENERAL_QUERY,
                routed_agent="error_handler",
                response_data={"error": str(e), "message": "An error occurred processing your request"},
                processing_time_ms=processing_time,
                confidence=0.0
            )
    
    def _classify_intent(self, query: str) -> tuple[UserIntent, float]:
        """Classify user intent using pattern matching"""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    confidence = 0.8 + random.random() * 0.2  # Mock confidence 0.8-1.0
                    return intent, confidence
        
        return UserIntent.GENERAL_QUERY, 0.5
    
    def _route_to_agent(self, intent: UserIntent, query: str) -> tuple[str, Union[TrendingNewsResponse, FactCheckResponse, SummarizeResponse, Dict]]:
        """Route query to appropriate specialist agent"""
        
        if intent == UserIntent.GET_TRENDING:
            # Extract topic and category from query
            topic = self._extract_topic(query)
            category = self._extract_category(query)
            
            request = TrendingNewsRequest(topic=topic, category=category)
            response = self.trending_agent.get_trending_news(request)
            return "TrendingNewsAgent", response
            
        elif intent == UserIntent.VERIFY_CLAIM:
            # Extract claim from query
            claim = self._extract_claim(query)
            
            request = FactCheckRequest(claim=claim)
            response = self.fact_checker_agent.fact_check_claim(request)
            return "FactCheckerAgent", response
            
        elif intent == UserIntent.SUMMARIZE_NEWS:
            # For demo, use a sample article - in real system, user would provide text
            sample_article = self._get_sample_article()
            
            request = SummarizeRequest(article_text=sample_article)
            response = self.summarizer_agent.summarize_news(request)
            return "NewsSummarizerAgent", response
            
        else:
            # General query - provide help message
            help_response = {
                "message": "I can help you with:",
                "capabilities": [
                    "Get trending news: 'What's trending in tech?'",
                    "Fact-check claims: 'Is it true that Apple acquired OpenAI?'", 
                    "Summarize articles: 'Summarize this article' (then provide text)"
                ],
                "examples": [
                    "Show me latest tech news",
                    "Fact check: Did Meta release a new AI model?",
                    "Summarize the latest AI developments"
                ]
            }
            return "GeneralHelpHandler", help_response
    
    def _extract_topic(self, query: str) -> str:
        """Extract topic from trending news query"""
        # Simple extraction - look for topic keywords
        topics = ["tech", "ai", "politics", "finance", "sports", "entertainment", "science", "health"]
        query_lower = query.lower()
        
        for topic in topics:
            if topic in query_lower:
                return topic
        
        return "general news"
    
    def _extract_category(self, query: str) -> NewsCategory:
        """Extract category from query"""
        query_lower = query.lower()
        
        category_mapping = {
            "tech": NewsCategory.TECH,
            "technology": NewsCategory.TECH,
            "ai": NewsCategory.TECH,
            "politics": NewsCategory.POLITICS,
            "finance": NewsCategory.FINANCE,
            "money": NewsCategory.FINANCE,
            "sports": NewsCategory.SPORTS,
            "entertainment": NewsCategory.ENTERTAINMENT,
            "science": NewsCategory.SCIENCE,
            "health": NewsCategory.HEALTH
        }
        
        for keyword, category in category_mapping.items():
            if keyword in query_lower:
                return category
        
        return NewsCategory.GENERAL
    
    def _extract_claim(self, query: str) -> str:
        """Extract claim from fact-check query"""
        # Remove question words and fact-check indicators
        clean_patterns = [
            r'\b(is it true that|verify that|fact check|check if|did really|true or false)\b',
            r'[\?!]+$'
        ]
        
        claim = query
        for pattern in clean_patterns:
            claim = re.sub(pattern, '', claim, flags=re.IGNORECASE)
        
        return claim.strip()
    
    def _get_sample_article(self) -> str:
        """Return sample article for summarization demo"""
        return """
        Meta has announced the release of their latest artificial intelligence model, which the company claims surpasses GPT-4 in several key benchmarks. The new model, called Meta AI Advanced, represents a significant leap forward in natural language processing capabilities.
        
        According to Meta's research team, the model shows improved performance in reasoning tasks, code generation, and multilingual understanding. The company conducted extensive testing across various domains including mathematics, science, and creative writing.
        
        The announcement comes amid fierce competition in the AI space, with companies like OpenAI, Google, and Anthropic all racing to develop more sophisticated language models. Meta's new model is expected to be integrated into their existing products including Facebook, Instagram, and WhatsApp.
        
        Industry experts are calling this development a game-changer for the AI landscape. The model's ability to understand context and generate human-like responses could revolutionize how people interact with digital assistants and content creation tools.
        
        Meta plans to make the model available through their developer API starting next quarter, with pricing details to be announced soon. The company also emphasized their commitment to responsible AI development and safety measures.
        """

# Main NewsSense System
class NewsSenseSystem:
    """Main system orchestrating all agents"""
    
    def __init__(self):
        self.conversation_agent = ConversationAgent()
        logfire.info("NewsSense System: Initialized successfully")
    
    def process_query(self, user_query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        """Main entry point for processing user queries"""
        logfire.info("NewsSense System: Processing query", user_query=user_query[:100], user_id=user_id)
        
        request = ConversationRequest(
            user_query=user_query,
            user_id=user_id,
            session_id=session_id
        )
        
        response = self.conversation_agent.process_conversation(request)
        
        # Convert to dictionary for easy JSON serialization
        return {
            "intent": response.intent.value,
            "routed_agent": response.routed_agent,
            "response_data": self._serialize_response_data(response.response_data),
            "processing_time_ms": response.processing_time_ms,
            "confidence": response.confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    def _serialize_response_data(self, response_data) -> Dict:
        """Convert Pydantic models to dictionaries for serialization"""
        if isinstance(response_data, BaseModel):
            return response_data.dict()
        return response_data
    
    def get_system_logs(self) -> List[Dict]:
        """Retrieve all system logs"""
        return logfire.get_logs()

# Demo and Testing Functions
def run_demo():
    """Run demonstration of the NewsSense system"""
    print("🚀 Starting NewsSense System Demo")
    print("=" * 50)
    
    # Initialize system
    news_system = NewsSenseSystem()
    
    # Test queries
    test_queries = [
        "What's trending in AI today?",
        "Is it true that Apple is partnering with OpenAI?",
        "Summarize the latest tech developments",
        "Show me recent finance news",
        "Fact check: Did Meta release a new AI model?",
        "Help me understand what you can do"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 30)
        
        try:
            result = news_system.process_query(query, user_id=f"demo_user_{i}")
            
            print(f"Intent: {result['intent']}")
            print(f"Routed to: {result['routed_agent']}")
            print(f"Processing time: {result['processing_time_ms']:.1f}ms")
            print(f"Confidence: {result['confidence']:.2f}")
            
            # Display response based on agent type
            response_data = result['response_data']
            
            if result['routed_agent'] == 'TrendingNewsAgent':
                headlines = response_data.get('headlines', [])
                print(f"Found {len(headlines)} trending headlines:")
                for headline in headlines[:3]:  # Show top 3
                    print(f"  • {headline['title']} ({headline['source']})")
            
            elif result['routed_agent'] == 'FactCheckerAgent':
                print(f"Verdict: {response_data['verdict']}")
                print(f"Summary: {response_data['summary'][:100]}...")
            
            elif result['routed_agent'] == 'NewsSummarizerAgent':
                print("Summary Points:")
                for point in response_data['summary_points']:
                    print(f"  {point}")
            
            else:
                print(f"Response: {response_data.get('message', 'General response')}")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print("\n" + "=" * 50)
    print("📊 System Logs Summary:")
    logs = news_system.get_system_logs()
    print(f"Total log entries: {len(logs)}")
    
    log_levels = {}
    for log in logs:
        level = log['level']
        log_levels[level] = log_levels.get(level, 0) + 1
    
    for level, count in log_levels.items():
        print(f"  {level}: {count} entries")
    
    print("\n✅ Demo completed successfully!")
    return news_system

# Performance Testing
def run_performance_test():
    """Test system performance with multiple concurrent queries"""
    import time
    
    print("🏃‍♂️ Running Performance Test")
    print("=" * 30)
    
    news_system = NewsSenseSystem()
    test_queries = [
        "What's trending in tech?",
        "Fact check: Meta new AI model",
        "Summarize recent developments",
    ] * 10  # 30 total queries
    
    start_time = time.time()
    
    results = []
    for i, query in enumerate(test_queries):
        result = news_system.process_query(query, user_id=f"perf_test_{i}")
        results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Processed {len(test_queries)} queries in {total_time:.2f} seconds")
    print(f"Average response time: {total_time/len(test_queries):.3f} seconds")
    
    # Analyze response times
    processing_times = [r['processing_time_ms'] for r in results]
    avg_processing_time = sum(processing_times) / len(processing_times)
    max_processing_time = max(processing_times)
    min_processing_time = min(processing_times)
    
    print(f"Average processing time: {avg_processing_time:.1f}ms")
    print(f"Max processing time: {max_processing_time:.1f}ms") 
    print(f"Min processing time: {min_processing_time:.1f}ms")
    
    return results

if __name__ == "__main__":
    # Run the demo
    system = run_demo()
    
    print("\n" + "="*50)
    print("🔬 Running Performance Test")
    perf_results = run_performance_test()
    
    print("\n" + "="*50)
    print("🎯 NewsSense System Ready!")
    print("You can now interact with the system using:")
    print("  system.process_query('Your question here')")
