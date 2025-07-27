"""
Simple stub for web search enhancer
"""

class WebSearchEnhancer:
    """Stub web search enhancer"""
    def __init__(self):
        pass
    
    def enhance_query(self, query):
        return query

def is_search_query(query):
    """Simple function to detect if query is a search query"""
    search_keywords = ['search', 'find', 'lookup', 'what is', 'who is', 'where is', 'when is', 'how to']
    return any(keyword in query.lower() for keyword in search_keywords)
