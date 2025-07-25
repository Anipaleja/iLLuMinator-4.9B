"""
Enhanced Web Search System for iLLuMinator AI
Multi-source intelligent information retrieval with streamlined summarization
"""

import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Any
import re
import concurrent.futures
from urllib.parse import quote_plus
import json
from datetime import datetime

class EnhancedWebSearcher:
    """Advanced web search system with multiple data sources and intelligent summarization"""
    
    def __init__(self, ai_instance, max_results: int = 5):
        self.ai_instance = ai_instance
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Common stop words for better content filtering
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        ])
        
    def search_and_summarize(self, query: str) -> str:
        """Search multiple sources and create an intelligent, streamlined answer"""
        print(f"Enhanced search for: '{query}'")
        
        try:
            # Step 1: Try Wikipedia first (authoritative source)
            wiki_info = self._search_wikipedia(query)
            
            # Step 2: Get web search results
            web_info = self._search_web(query)
            
            # Step 3: Combine and create intelligent summary
            return self._create_streamlined_answer(query, wiki_info, web_info)
            
        except Exception as e:
            print(f"Enhanced search failed: {e}")
            return f"I encountered an error while searching: {str(e)}"
    
    def _search_wikipedia(self, query: str) -> Dict[str, Any]:
        """Search Wikipedia for authoritative information"""
        try:
            import wikipedia
            wikipedia.set_lang("en")
            
            # Search for relevant articles
            search_results = wikipedia.search(query, results=2)
            
            if not search_results:
                return None
            
            # Get the most relevant article
            try:
                page = wikipedia.page(search_results[0])
                return {
                    'title': page.title,
                    'summary': page.summary,
                    'url': page.url,
                    'source': 'Wikipedia'
                }
            except wikipedia.exceptions.DisambiguationError as e:
                # Try the first option from disambiguation
                page = wikipedia.page(e.options[0])
                return {
                    'title': page.title,
                    'summary': page.summary,
                    'url': page.url,
                    'source': 'Wikipedia'
                }
        except Exception as e:
            print(f"Wikipedia search failed: {e}")
            return None
    
    def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced web search with content extraction"""
        urls = self._search_ddg(query)
        if not urls:
            return []
        
        # Scrape content from top URLs
        web_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {executor.submit(self._scrape_url, url): url for url in urls[:3]}
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    result = future.result()
                    if result:
                        web_results.append(result)
                except Exception as e:
                    print(f"Error scraping {future_to_url[future]}: {e}")
        
        return web_results
    
    def _search_ddg(self, query: str) -> List[str]:
        """Search DuckDuckGo and return top URLs"""
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        try:
            response = self.session.get(search_url, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', class_='result__a')
            
            urls = [link['href'] for link in links]
            
            # Filter out unwanted URLs
            urls = [
                url for url in urls 
                if not url.startswith("https://duckduckgo.com/y.js") 
                and url.startswith("http")
            ]
            
            return urls[:self.max_results]
            
        except requests.RequestException as e:
            print(f"DuckDuckGo search failed: {e}")
            return []
    
    def _scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape and extract meaningful content from URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "ads"]):
                element.extract()
            
            # Try to find main content
            main_content = (soup.find('main') or 
                          soup.find('article') or 
                          soup.find('div', class_='content') or 
                          soup.find('div', id='content') or
                          soup)
            
            # Extract text
            text = main_content.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk) > 15)
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "Web Page"
            
            return {
                'title': title[:100],  # Limit title length
                'content': clean_text[:1500],  # Limit content
                'url': url,
                'source': 'Web'
            }
            
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            return None
    
    def _create_streamlined_answer(self, query: str, wiki_info: Dict, web_info: List[Dict]) -> str:
        """Create a streamlined, intelligent answer from all sources"""
        
        # Extract key information
        key_facts = []
        sources = []
        
        # Process Wikipedia (highest priority)
        if wiki_info:
            wiki_facts = self._extract_key_sentences(query, wiki_info['summary'])
            key_facts.extend([(fact, 'Wikipedia', 3) for fact in wiki_facts])
            sources.append('Wikipedia')
        
        # Process web results
        for web_item in web_info:
            if web_item and web_item.get('content'):
                web_facts = self._extract_key_sentences(query, web_item['content'])
                weight = 2 if 'news' in web_item.get('title', '').lower() else 1
                key_facts.extend([(fact, web_item.get('title', 'Web'), weight) for fact in web_facts])
                sources.append('Web')
        
        if not key_facts:
            # Better fallback for when no results found
            return self._create_fallback_response(query)
        
        # Rank facts by relevance and authority
        ranked_facts = self._rank_and_deduplicate(query, key_facts)
        
        # Create final answer
        answer = self._synthesize_final_answer(query, ranked_facts)
        
        # Add source attribution
        unique_sources = list(set(sources))
        if unique_sources:
            answer += f"\n\n*Information from: {', '.join(unique_sources)}*"
        
        return answer
    
    def _create_fallback_response(self, query: str) -> str:
        """Create a helpful fallback response when no web results are found"""
        query_lower = query.lower()
        
        # Check if it's a current events query
        if any(word in query_lower for word in ['latest', 'current', 'recent', 'news', 'today']):
            return "I wasn't able to find current information about this topic from my available sources. You might want to check recent news websites or official sources for the most up-to-date information."
        
        # Check if it's a weather query
        if 'weather' in query_lower:
            return "I don't have access to real-time weather data. For current weather information, I'd recommend checking weather.com, your local weather app, or asking a voice assistant with weather access."
        
        # Check if it's a location-specific query
        if any(word in query_lower for word in ['where is', 'location of', 'address of']):
            return "I couldn't find specific location information for this query. You might want to try Google Maps or other mapping services for detailed location data."
        
        # General fallback
        return "I searched for information about this topic but couldn't find relevant details from available sources. You might want to try rephrasing your question or checking specialized sources for this particular topic."
    
    def _extract_key_sentences(self, query: str, content: str) -> List[str]:
        """Extract the most relevant sentences from content"""
        if not content:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        query_words = set(word.lower() for word in query.split() if word.lower() not in self.stop_words)
        
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30 or len(sentence) > 300:  # Skip too short/long
                continue
            
            sentence_words = set(word.lower() for word in re.findall(r'\b\w+\b', sentence))
            overlap = query_words.intersection(sentence_words)
            
            if overlap and len(overlap) / len(query_words) > 0.25:  # At least 25% overlap
                relevant_sentences.append(sentence)
        
        return relevant_sentences[:5]  # Top 5 sentences
    
    def _rank_and_deduplicate(self, query: str, facts: List[tuple]) -> List[tuple]:
        """Rank facts by relevance and remove duplicates"""
        query_words = set(word.lower() for word in query.split())
        
        # Score each fact
        scored_facts = []
        for fact, source, weight in facts:
            fact_words = set(word.lower() for word in re.findall(r'\b\w+\b', fact))
            relevance = len(query_words.intersection(fact_words)) / max(len(query_words), 1)
            score = relevance * weight
            scored_facts.append((fact, source, score))
        
        # Sort by score and remove similar facts
        scored_facts.sort(key=lambda x: x[2], reverse=True)
        
        # Deduplicate similar sentences
        unique_facts = []
        for fact, source, score in scored_facts:
            if not any(self._are_similar(fact, existing[0]) for existing in unique_facts):
                unique_facts.append((fact, source, score))
        
        return unique_facts[:4]  # Top 4 unique facts
    
    def _are_similar(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are similar"""
        words1 = set(word.lower() for word in re.findall(r'\b\w+\b', sent1))
        words2 = set(word.lower() for word in re.findall(r'\b\w+\b', sent2))
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total > 0.7  # 70% similarity threshold
    
    def _synthesize_final_answer(self, query: str, ranked_facts: List[tuple]) -> str:
        """Create a coherent, natural answer from ranked facts"""
        if not ranked_facts:
            return "I couldn't find specific information to answer your question."
        
        query_lower = query.lower()
        
        # Determine answer style based on question type
        if any(word in query_lower for word in ['what is', 'what are', 'define', 'meaning']):
            return self._create_definition_style(ranked_facts)
        elif any(word in query_lower for word in ['how', 'why', 'explain']):
            return self._create_explanation_style(ranked_facts)
        elif any(word in query_lower for word in ['when', 'where', 'who']):
            return self._create_factual_style(ranked_facts)
        elif any(word in query_lower for word in ['latest', 'recent', 'news', 'current']):
            return self._create_current_events_style(ranked_facts)
        else:
            return self._create_general_style(ranked_facts)
    
    def _create_definition_style(self, facts: List[tuple]) -> str:
        """Create a definition-style answer"""
        main_fact = facts[0][0]
        
        if len(facts) > 1:
            supporting = facts[1][0]
            return f"{main_fact} {supporting}"
        
        return main_fact
    
    def _create_explanation_style(self, facts: List[tuple]) -> str:
        """Create an explanatory answer"""
        answer_parts = [fact[0] for fact in facts[:2]]
        return ' '.join(answer_parts)
    
    def _create_factual_style(self, facts: List[tuple]) -> str:
        """Create a factual answer"""
        return facts[0][0]
    
    def _create_current_events_style(self, facts: List[tuple]) -> str:
        """Create a current events answer"""
        if len(facts) >= 2:
            return f"{facts[0][0]} {facts[1][0]}"
        return facts[0][0]
    
    def _create_general_style(self, facts: List[tuple]) -> str:
        """Create a general informational answer"""
        return facts[0][0]


def is_search_query(query: str) -> bool:
    """Enhanced query detection for web search"""
    query_lower = query.lower().strip()
    
    # Skip very simple greetings and conversational phrases
    simple_patterns = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'thanks', 'thank you', 'bye', 'goodbye',
        'yes', 'no', 'ok', 'okay', 'sure'
    ]
    
    if any(query_lower == pattern or query_lower.startswith(pattern + ' ') for pattern in simple_patterns):
        return False
    
    # Strong indicators for web search
    search_indicators = [
        'who is', 'what is', 'when did', 'where is', 'why do', 'how to',
        'search for', 'find information', 'tell me about',
        'latest news', 'current status', 'what happened',
        'recent updates', 'latest information', 'breaking news'
    ]
    
    if any(query_lower.startswith(indicator) for indicator in search_indicators):
        return True
    
    # Check for factual questions
    if query.endswith('?'):
        factual_words = ['what', 'when', 'where', 'who', 'which', 'price', 'cost', 
                        'weather', 'temperature', 'latest', 'current', 'recent', 'news']
        
        if any(word in query_lower for word in factual_words):
            # Skip basic programming questions unless asking for current info
            programming_terms = ['function', 'code', 'program', 'python', 'javascript', 'algorithm']
            if any(term in query_lower for term in programming_terms):
                current_terms = ['latest', 'current', 'recent', 'news', 'update']
                return any(term in query_lower for term in current_terms)
            return True
    
    return False


if __name__ == '__main__':
    # Test the enhanced search system
    class MockAI:
        pass
    
    print("Testing Enhanced Web Search System...")
    
    searcher = EnhancedWebSearcher(MockAI())
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "Who is the current president of France?",
        "Latest news about climate change",
        "How does photosynthesis work?"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        print(f"Should search: {is_search_query(query)}")
        # Uncomment to test actual search
        # result = searcher.search_and_summarize(query)
        # print(f"Result: {result[:200]}...")
