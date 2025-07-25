"""
Web Search Enhancer for iLLuMinator AI
Dynamically fetches and summarizes web content to answer user queries
"""

import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Any
import re
import concurrent.futures
from urllib.parse import quote_plus

class WebSearchEnhancer:
    """Enhances AI responses with real-time web search results"""
    
    def __init__(self, ai_instance, max_results: int = 3):
        self.ai_instance = ai_instance
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def search_and_summarize(self, query: str) -> str:
        """Search the web for a query, scrape results, and summarize them"""
        print(f"Performing web search for: '{query}'")
        
        try:
            # Get search results
            urls = self._search_ddg(query)
            if not urls:
                return "I couldn't find any relevant information online."
            
            # Scrape and summarize content
            print(f"Scraping {len(urls)} URLs...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_results) as executor:
                future_to_url = {executor.submit(self._scrape_url, url): url for url in urls}
                contents = []
                for future in concurrent.futures.as_completed(future_to_url):
                    try:
                        content = future.result()
                        if content:
                            contents.append(content)
                    except Exception as e:
                        print(f"Error scraping {future_to_url[future]}: {e}")
            
            if not contents:
                return "I found some web pages, but I was unable to extract their content."
            
            # Combine and summarize
            full_content = "\n\n".join(contents)
            summary = self._summarize_content(query, full_content)
            
            return summary
            
        except Exception as e:
            print(f"Web search failed: {e}")
            return f"I encountered an error while searching the web: {e}"
            
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
            
    def _scrape_url(self, url: str) -> str:
        """Scrape text content from a single URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text and clean it
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            return text[:3000]
            
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            return ""
            
    def _summarize_content(self, query: str, content: str) -> str:
        """Summarize content based on web search results"""
        print("Summarizing web content...")
        
        # Simple extraction-based summarization instead of using AI generation
        if not content or len(content) < 50:
            return "I found some web pages, but they didn't contain enough relevant information to answer your question."
        
        # Extract key sentences that might be relevant to the query
        query_words = set(query.lower().split())
        sentences = content.replace('\n', ' ').split('.')
        
        relevant_sentences = []
        for sentence in sentences[:20]:  # Check first 20 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                sentence_words = set(sentence.lower().split())
                # Check if sentence contains query keywords
                if query_words.intersection(sentence_words) or any(word in sentence.lower() for word in query_words):
                    relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Return the most relevant sentences as a summary
            summary = '. '.join(relevant_sentences[:3]) + '.'
            return f"Based on web search results: {summary}"
        else:
            # Fallback to first part of content
            first_part = content[:500].strip()
            if first_part:
                return f"Here's what I found online: {first_part}..."
            else:
                return "I searched the web but couldn't find relevant information to answer your question."

def is_search_query(query: str) -> bool:
    """Check if a query is likely a web search request"""
    query_lower = query.lower().strip()
    
    # Skip very simple greetings and conversational phrases
    simple_patterns = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'thanks', 'thank you', 'bye', 'goodbye',
        'yes', 'no', 'ok', 'okay', 'sure'
    ]
    
    if any(query_lower == pattern or query_lower.startswith(pattern + ' ') for pattern in simple_patterns):
        return False
    
    # Keywords that definitely suggest a web search
    search_keywords = [
        'who is', 'what is', 'when did', 'where is', 'why do', 'how to',
        'search for', 'find information about', 'tell me about',
        'latest news on', 'current status of', 'what happened to',
        'recent updates on', 'latest information about'
    ]
    
    # Check if it starts with search keywords
    if any(query_lower.startswith(keyword) for keyword in search_keywords):
        return True
    
    # More selective question detection - must have specific indicators
    if query.endswith('?'):
        # Check for current events, factual questions, or specific topics
        current_indicators = ['latest', 'current', 'recent', 'today', 'now', 'news', 'update']
        factual_indicators = ['what', 'when', 'where', 'who', 'which', 'price', 'cost', 'weather', 'temperature']
        
        if any(indicator in query_lower for indicator in current_indicators + factual_indicators):
            # But skip programming-related questions unless they ask for current info
            programming_terms = ['function', 'code', 'program', 'python', 'javascript', 'programming', 'algorithm']
            if any(term in query_lower for term in programming_terms):
                return any(indicator in query_lower for indicator in current_indicators)
            return True
    
    # Check for other specific search indicators
    search_indicators = ['news about', 'information on', 'details about', 'facts about']
    if any(indicator in query_lower for indicator in search_indicators):
        return True
    
    return False

if __name__ == '__main__':
    # Example usage (requires a mock AI instance)
    class MockAI:
        def generate_response(self, prompt, max_tokens, temperature):
            return f"This is a summary for the query based on the provided content. It would normally be generated by the AI model. The original query was about various topics, and this summary synthesizes the key points into a coherent answer."

    print("Testing Web Search Enhancer...")
    ai_mock = MockAI()
    search_enhancer = WebSearchEnhancer(ai_instance=ai_mock)
    
    test_query = "What is the latest news on Python 4.0?"
    summary = search_enhancer.search_and_summarize(test_query)
    
    print(f"\nQuery: {test_query}")
    print(f"Summary: {summary}")
    
    print("\nTesting query detection:")
    print(f"'search for python decorators' -> {is_search_query('search for python decorators')}")
    print(f"'how are you?' -> {is_search_query('how are you?')}")
    print(f"'what is the capital of france?' -> {is_search_query('what is the capital of france?')}")
    print(f"'write a python function' -> {is_search_query('write a python function')}")
