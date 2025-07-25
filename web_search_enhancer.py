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
import json
from datetime import datetime
import nltk
from collections import Counter

class WebSearchEnhancer:
    """Enhances AI responses with real-time web search results"""
    
    def __init__(self, ai_instance, max_results: int = 5):
        self.ai_instance = ai_instance
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize data sources
        self.data_sources = {
            'wikipedia': True,
            'news': True,
            'web_search': True,
            'academic': True
        }
        
        # Common stop words for better content filtering
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        ])
        
    def search_and_summarize(self, query: str) -> str:
        """Search multiple data sources and create a streamlined, intelligent summary"""
        print(f"Performing multi-source search for: '{query}'")
        
        try:
            # Gather information from multiple sources
            sources_data = self._gather_from_sources(query)
            
            if not sources_data:
                return "I couldn't find any relevant information from available sources."
            
            # Create an intelligent, streamlined summary
            summary = self._create_intelligent_summary(query, sources_data)
            
            return summary
            
        except Exception as e:
            print(f"Multi-source search failed: {e}")
            return f"I encountered an error while searching for information: {str(e)}"
    
    def _gather_from_sources(self, query: str) -> Dict[str, Any]:
        """Gather information from multiple high-quality sources"""
        sources_data = {}
        
        # Try Wikipedia first (often has authoritative, well-structured info)
        if self.data_sources['wikipedia']:
            wiki_data = self._search_wikipedia(query)
            if wiki_data:
                sources_data['wikipedia'] = wiki_data
        
        # Try web search for current/breaking information
        if self.data_sources['web_search']:
            web_data = self._search_web_enhanced(query)
            if web_data:
                sources_data['web'] = web_data
        
        # Try news sources for current events
        if self.data_sources['news'] and self._is_news_query(query):
            news_data = self._search_news_sources(query)
            if news_data:
                sources_data['news'] = news_data
        
        return sources_data
    def _search_wikipedia(self, query: str) -> Dict[str, Any]:
        """Search Wikipedia for authoritative information"""
        try:
            import wikipedia
            wikipedia.set_lang("en")
            
            # Search for relevant articles
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return None
            
            # Get the most relevant article
            try:
                page = wikipedia.page(search_results[0])
                return {
                    'title': page.title,
                    'summary': page.summary[:800],  # First 800 chars
                    'content': page.content[:2000],  # First 2000 chars
                    'url': page.url,
                    'source': 'Wikipedia'
                }
            except wikipedia.exceptions.DisambiguationError as e:
                # Try the first option from disambiguation
                page = wikipedia.page(e.options[0])
                return {
                    'title': page.title,
                    'summary': page.summary[:800],
                    'content': page.content[:2000],
                    'url': page.url,
                    'source': 'Wikipedia'
                }
        except Exception as e:
            print(f"Wikipedia search failed: {e}")
            return None
    
    def _search_web_enhanced(self, query: str) -> Dict[str, Any]:
        """Enhanced web search with better content extraction"""
        urls = self._search_ddg(query)
        if not urls:
            return None
        
        # Scrape content from top URLs
        contents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(urls))) as executor:
            future_to_url = {executor.submit(self._scrape_url_enhanced, url): url for url in urls[:3]}
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    content = future.result()
                    if content:
                        contents.append(content)
                except Exception as e:
                    print(f"Error scraping {future_to_url[future]}: {e}")
        
        if contents:
            return {
                'contents': contents,
                'source': 'Web Search'
            }
        return None
    
    def _search_news_sources(self, query: str) -> Dict[str, Any]:
        """Search news sources for current information"""
        # For now, use web search with news-specific terms
        news_query = f"{query} news recent latest"
        return self._search_web_enhanced(news_query)
    
    def _is_news_query(self, query: str) -> bool:
        """Check if query is asking for news/current events"""
        news_keywords = ['news', 'latest', 'recent', 'today', 'current', 'breaking', 'update', 'happened']
        return any(keyword in query.lower() for keyword in news_keywords)
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
            
    def _scrape_url_enhanced(self, url: str) -> Dict[str, Any]:
        """Enhanced URL scraping with better content extraction"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.extract()
            
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if not main_content:
                main_content = soup
            
            # Get text and clean it
            text = main_content.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk) > 10)
            
            # Extract title
            title = soup.find('title').get_text() if soup.find('title') else url
            
            return {
                'title': title.strip(),
                'content': text[:2000],  # Limit content
                'url': url,
                'source': 'Web'
            }
            
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            return None
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
            
    def _create_intelligent_summary(self, query: str, sources_data: Dict[str, Any]) -> str:
        """Create a streamlined, intelligent summary from multiple sources"""
        print("Creating intelligent summary...")
        
        # Extract key information from all sources
        all_facts = []
        source_names = []
        
        # Process Wikipedia data (most authoritative)
        if 'wikipedia' in sources_data:
            wiki = sources_data['wikipedia']
            facts = self._extract_key_facts(query, wiki['summary'])
            all_facts.extend([(fact, 'Wikipedia', 3) for fact in facts])  # Higher weight
            source_names.append('Wikipedia')
        
        # Process web search data
        if 'web' in sources_data:
            for content in sources_data['web']['contents']:
                if content:
                    facts = self._extract_key_facts(query, content['content'])
                    all_facts.extend([(fact, content.get('title', 'Web'), 2) for fact in facts])
            source_names.append('Web sources')
        
        # Process news data
        if 'news' in sources_data:
            for content in sources_data['news']['contents']:
                if content:
                    facts = self._extract_key_facts(query, content['content'])
                    all_facts.extend([(fact, content.get('title', 'News'), 2) for fact in facts])
            source_names.append('News sources')
        
        if not all_facts:
            return "I found some sources but couldn't extract relevant information to answer your question."
        
        # Rank and select the best facts
        ranked_facts = self._rank_facts(query, all_facts)
        
        # Create a coherent, streamlined response
        summary = self._synthesize_answer(query, ranked_facts[:5])  # Top 5 facts
        
        # Add source attribution
        if source_names:
            sources_text = ', '.join(list(set(source_names)))
            summary += f"\n\n*Sources: {sources_text}*"
        
        return summary
    
    def _extract_key_facts(self, query: str, content: str) -> List[str]:
        """Extract key facts from content that are relevant to the query"""
        if not content:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        # Get query keywords
        query_words = set(word.lower() for word in query.split() if word.lower() not in self.stop_words)
        
        relevant_facts = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 200:  # Skip too short/long sentences
                continue
            
            sentence_words = set(word.lower() for word in re.findall(r'\b\w+\b', sentence))
            
            # Calculate relevance score
            common_words = query_words.intersection(sentence_words)
            if common_words:
                relevance = len(common_words) / len(query_words)
                if relevance > 0.2:  # At least 20% keyword overlap
                    relevant_facts.append(sentence.strip())
        
        return relevant_facts[:10]  # Top 10 facts
    
    def _rank_facts(self, query: str, facts: List[tuple]) -> List[tuple]:
        """Rank facts by relevance and source authority"""
        query_words = set(word.lower() for word in query.split())
        
        scored_facts = []
        for fact, source, weight in facts:
            fact_words = set(word.lower() for word in re.findall(r'\b\w+\b', fact))
            
            # Calculate relevance score
            common_words = query_words.intersection(fact_words)
            relevance = len(common_words) / max(len(query_words), 1)
            
            # Calculate final score (relevance * source weight)
            score = relevance * weight
            scored_facts.append((fact, source, weight, score))
        
        # Sort by score descending
        return sorted(scored_facts, key=lambda x: x[3], reverse=True)
    
    def _synthesize_answer(self, query: str, ranked_facts: List[tuple]) -> str:
        """Synthesize ranked facts into a coherent, streamlined answer"""
        if not ranked_facts:
            return "I couldn't find specific information to answer your question."
        
        # Determine the type of question to structure the answer appropriately
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'what are', 'define']):
            # Definition-style answer
            answer = self._create_definition_answer(ranked_facts)
        elif any(word in query_lower for word in ['how', 'why', 'when', 'where']):
            # Explanatory answer
            answer = self._create_explanatory_answer(ranked_facts)
        elif any(word in query_lower for word in ['latest', 'recent', 'news', 'current']):
            # Current events answer
            answer = self._create_current_events_answer(ranked_facts)
        else:
            # General informational answer
            answer = self._create_general_answer(ranked_facts)
        
        return answer
    
    def _create_definition_answer(self, facts: List[tuple]) -> str:
        """Create a definition-style answer"""
        top_fact, _, _, _ = facts[0]
        answer = top_fact
        
        if len(facts) > 1:
            # Add supporting information
            supporting = facts[1][0]
            if not self._sentences_similar(answer, supporting):
                answer += f" {supporting}"
        
        return answer
    
    def _create_explanatory_answer(self, facts: List[tuple]) -> str:
        """Create an explanatory answer"""
        answer_parts = []
        
        for fact, _, _, _ in facts[:3]:  # Use top 3 facts
            if not any(self._sentences_similar(fact, existing) for existing in answer_parts):
                answer_parts.append(fact)
        
        return ' '.join(answer_parts)
    
    def _create_current_events_answer(self, facts: List[tuple]) -> str:
        """Create a current events answer"""
        answer_parts = []
        
        for fact, _, _, _ in facts[:2]:  # Use top 2 facts for news
            if not any(self._sentences_similar(fact, existing) for existing in answer_parts):
                answer_parts.append(fact)
        
        if answer_parts:
            return ' '.join(answer_parts)
        return "I found some recent information but couldn't extract clear details."
    
    def _create_general_answer(self, facts: List[tuple]) -> str:
        """Create a general informational answer"""
        top_fact, _, _, _ = facts[0]
        return top_fact
    
    def _sentences_similar(self, sent1: str, sent2: str, threshold: float = 0.6) -> bool:
        """Check if two sentences are similar to avoid repetition"""
        words1 = set(word.lower() for word in re.findall(r'\b\w+\b', sent1))
        words2 = set(word.lower() for word in re.findall(r'\b\w+\b', sent2))
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union > threshold
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
