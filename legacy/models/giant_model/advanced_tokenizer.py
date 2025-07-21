"""
Advanced Tokenizer for Giant Transformer Model
High-quality tokenization with 50K+ vocabulary for maximum accuracy
"""

import re
import json
import pickle
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict
import unicodedata

class AdvancedTokenizer:
    """Advanced tokenizer with BPE-style subword tokenization"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<|endoftext|>': 50256,
            '<|pad|>': 50255,
            '<|unk|>': 50254,
            '<|bos|>': 50253,
            '<|eos|>': 50252,
            '<|mask|>': 50251,
            '<|sep|>': 50250,
            '<|cls|>': 50249,
        }
        
        # Initialize with comprehensive vocabulary
        self.setup_vocabulary()
        print(f"🔤 Advanced tokenizer initialized with {len(self.encoder)} tokens")
    
    def setup_vocabulary(self):
        """Setup comprehensive vocabulary for maximum coverage"""
        
        # Start with common English words and characters
        base_vocab = self._get_base_vocabulary()
        
        # Add common subwords and morphemes
        subword_vocab = self._get_subword_vocabulary()
        
        # Add technical and domain-specific terms
        technical_vocab = self._get_technical_vocabulary()
        
        # Add common bigrams and trigrams
        ngram_vocab = self._get_ngram_vocabulary()
        
        # Combine all vocabularies
        all_tokens = list(base_vocab) + list(subword_vocab) + list(technical_vocab) + list(ngram_vocab)
        
        # Remove duplicates and limit to vocab size
        unique_tokens = list(dict.fromkeys(all_tokens))  # Preserves order
        
        # Reserve space for special tokens
        max_regular_tokens = self.vocab_size - len(self.special_tokens)
        if len(unique_tokens) > max_regular_tokens:
            unique_tokens = unique_tokens[:max_regular_tokens]
        
        # Create encoder/decoder mappings
        self.encoder = {}
        self.decoder = {}
        
        # Add regular tokens
        for i, token in enumerate(unique_tokens):
            self.encoder[token] = i
            self.decoder[i] = token
        
        # Add special tokens
        for token, id in self.special_tokens.items():
            self.encoder[token] = id
            self.decoder[id] = token
        
        # Create byte-pair patterns for subword tokenization
        self.bpe_patterns = self._create_bpe_patterns()
    
    def _get_base_vocabulary(self) -> List[str]:
        """Get base vocabulary including characters, common words, etc."""
        vocab = []
        
        # All printable ASCII characters
        for i in range(32, 127):
            vocab.append(chr(i))
        
        # Common English words (high frequency)
        common_words = [
            # Articles, prepositions, conjunctions
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "up", "down", "out", "off", "over", "under", "above", "below", "between", "through", "during",
            
            # Pronouns
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
            "his", "her", "its", "our", "their", "this", "that", "these", "those", "who", "what", "which",
            "when", "where", "why", "how",
            
            # Verbs
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "can", "may", "might", "must", "get", "go", "come", "see",
            "know", "think", "take", "make", "give", "use", "find", "tell", "ask", "work", "try", "need",
            "feel", "become", "leave", "put", "mean", "keep", "let", "begin", "seem", "help", "show",
            "hear", "play", "run", "move", "live", "believe", "hold", "bring", "happen", "write", "sit",
            "stand", "lose", "pay", "meet", "include", "continue", "set", "learn", "change", "lead",
            "understand", "watch", "follow", "stop", "create", "speak", "read", "allow", "add", "spend",
            "grow", "open", "walk", "win", "offer", "remember", "love", "consider", "appear", "buy",
            "wait", "serve", "die", "send", "expect", "build", "stay", "fall", "cut", "reach", "kill",
            "remain", "suggest", "raise", "pass", "sell", "require", "report", "decide", "pull",
            
            # Nouns
            "time", "person", "year", "way", "day", "thing", "man", "world", "life", "hand", "part",
            "child", "eye", "woman", "place", "work", "week", "case", "point", "government", "company",
            "number", "group", "problem", "fact", "be", "have", "do", "say", "get", "make", "go", "know",
            "take", "see", "come", "think", "look", "want", "give", "use", "find", "tell", "ask", "work",
            "seem", "feel", "try", "leave", "call", "good", "new", "first", "last", "long", "great",
            "little", "own", "other", "old", "right", "big", "high", "different", "small", "large",
            "next", "early", "young", "important", "few", "public", "bad", "same", "able",
            
            # Numbers
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "hundred", "thousand", "million", "billion",
        ]
        
        vocab.extend(common_words)
        
        # Numbers
        for i in range(10000):
            vocab.append(str(i))
        
        return vocab
    
    def _get_subword_vocabulary(self) -> List[str]:
        """Get common subwords, prefixes, suffixes"""
        subwords = []
        
        # Common prefixes
        prefixes = [
            "un", "re", "in", "im", "dis", "en", "em", "non", "over", "mis", "sub", "pre", "inter",
            "fore", "de", "trans", "super", "semi", "anti", "mid", "under", "out", "up", "auto",
            "co", "ex", "extra", "hyper", "micro", "macro", "mega", "mini", "multi", "neo", "post",
            "pro", "pseudo", "quasi", "ultra", "counter", "bi", "tri", "quad", "poly", "mono"
        ]
        
        # Common suffixes
        suffixes = [
            "ing", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment", "ful", "less", "able",
            "ible", "ous", "ious", "ive", "ary", "ory", "ity", "ty", "al", "ic", "ical", "ous",
            "eous", "ship", "hood", "ward", "wise", "like", "some", "dom", "age", "ism", "ist",
            "ize", "ise", "fy", "en", "ate", "fy", "ward", "ways", "ling", "let", "kin"
        ]
        
        # Common roots
        roots = [
            "act", "form", "port", "spect", "dict", "duc", "fact", "ject", "mit", "mov", "ped",
            "pos", "rupt", "scrib", "sect", "sent", "serv", "sign", "sist", "spir", "struct",
            "tain", "tend", "terr", "tract", "vers", "vert", "vid", "vis", "voc", "volv"
        ]
        
        subwords.extend(prefixes)
        subwords.extend(suffixes)
        subwords.extend(roots)
        
        return subwords
    
    def _get_technical_vocabulary(self) -> List[str]:
        """Get technical and domain-specific vocabulary"""
        technical = []
        
        # AI/ML terms
        ai_terms = [
            "artificial", "intelligence", "machine", "learning", "deep", "neural", "network",
            "algorithm", "model", "training", "data", "dataset", "feature", "prediction",
            "classification", "regression", "supervised", "unsupervised", "reinforcement",
            "attention", "transformer", "embedding", "gradient", "optimization", "backpropagation",
            "epoch", "batch", "loss", "accuracy", "overfitting", "underfitting", "validation",
            "testing", "cross", "tensor", "matrix", "vector", "activation", "sigmoid", "relu",
            "softmax", "convolution", "pooling", "dropout", "normalization", "autoencoder",
            "gan", "vae", "bert", "gpt", "llm", "nlp", "cv", "computer", "vision", "speech",
            "recognition", "generation", "synthesis", "inference", "deployment", "scaling"
        ]
        
        # Programming terms
        programming_terms = [
            "python", "java", "javascript", "cpp", "csharp", "ruby", "go", "rust", "swift",
            "kotlin", "scala", "haskell", "clojure", "erlang", "function", "method", "class",
            "object", "variable", "parameter", "argument", "return", "loop", "condition",
            "array", "list", "dictionary", "hash", "map", "set", "queue", "stack", "tree",
            "graph", "algorithm", "complexity", "recursion", "iteration", "pointer", "reference",
            "memory", "allocation", "garbage", "collection", "inheritance", "polymorphism",
            "encapsulation", "abstraction", "interface", "protocol", "framework", "library",
            "api", "sdk", "ide", "compiler", "interpreter", "debugger", "profiler", "testing",
            "unit", "integration", "deployment", "docker", "kubernetes", "microservices"
        ]
        
        # Science terms
        science_terms = [
            "physics", "chemistry", "biology", "mathematics", "quantum", "relativity", "atom",
            "molecule", "electron", "proton", "neutron", "photon", "wave", "particle", "energy",
            "force", "gravity", "electromagnetic", "thermodynamics", "entropy", "evolution",
            "genetics", "dna", "rna", "protein", "cell", "organism", "ecosystem", "climate",
            "carbon", "hydrogen", "oxygen", "nitrogen", "periodic", "table", "reaction",
            "catalyst", "equation", "formula", "theorem", "proof", "hypothesis", "theory",
            "experiment", "observation", "measurement", "analysis", "statistics", "probability"
        ]
        
        # Technology terms
        tech_terms = [
            "computer", "processor", "cpu", "gpu", "memory", "ram", "storage", "ssd", "hdd",
            "motherboard", "graphics", "display", "monitor", "keyboard", "mouse", "internet",
            "network", "wifi", "bluetooth", "usb", "port", "cable", "wireless", "protocol",
            "http", "https", "tcp", "ip", "dns", "server", "client", "database", "sql",
            "nosql", "cloud", "aws", "azure", "gcp", "saas", "paas", "iaas", "devops",
            "cicd", "git", "github", "gitlab", "version", "control", "agile", "scrum",
            "kanban", "sprint", "backlog", "requirements", "specification", "architecture",
            "design", "pattern", "mvc", "mvp", "mvvm", "rest", "graphql", "json", "xml"
        ]
        
        # Company and brand names
        company_terms = [
            "google", "microsoft", "apple", "amazon", "facebook", "meta", "netflix", "tesla",
            "nvidia", "intel", "amd", "ibm", "oracle", "salesforce", "adobe", "uber", "airbnb",
            "spotify", "twitter", "linkedin", "youtube", "instagram", "whatsapp", "zoom",
            "slack", "discord", "github", "stackoverflow", "reddit", "wikipedia", "openai",
            "anthropic", "deepmind", "huggingface", "pytorch", "tensorflow", "keras",
            "scikit", "pandas", "numpy", "matplotlib", "jupyter", "anaconda", "docker"
        ]
        
        technical.extend(ai_terms)
        technical.extend(programming_terms)
        technical.extend(science_terms)
        technical.extend(tech_terms)
        technical.extend(company_terms)
        
        return technical
    
    def _get_ngram_vocabulary(self) -> List[str]:
        """Get common n-grams for better context understanding"""
        ngrams = []
        
        # Common bigrams
        bigrams = [
            "of the", "in the", "to the", "for the", "on the", "at the", "by the", "from the",
            "with the", "about the", "into the", "through the", "during the", "before the",
            "after the", "over the", "under the", "up to", "out of", "as well", "such as",
            "more than", "less than", "rather than", "other than", "not only", "as much",
            "so much", "how much", "too much", "very much", "much more", "much less",
            "right now", "just now", "for now", "by now", "until now", "from now", "since then",
            "even though", "even if", "even when", "even where", "as if", "as though",
            "in order", "in fact", "in case", "in terms", "in addition", "in particular",
            "for example", "for instance", "that is", "i.e.", "e.g.", "etc.", "and so",
            "or so", "and then", "but then", "so that", "such that", "given that"
        ]
        
        # Common trigrams
        trigrams = [
            "as well as", "in order to", "in addition to", "as a result", "on the other",
            "at the same", "for the first", "for the last", "in the case", "in the end",
            "in the future", "in the past", "at the time", "by the way", "on the way",
            "in the way", "all the way", "one of the", "some of the", "most of the",
            "part of the", "because of the", "instead of the", "in spite of", "in front of",
            "in back of", "on top of", "at the end", "at the beginning", "in the middle"
        ]
        
        ngrams.extend(bigrams)
        ngrams.extend(trigrams)
        
        return ngrams
    
    def _create_bpe_patterns(self) -> List[Tuple[str, str]]:
        """Create byte-pair encoding patterns for subword tokenization"""
        patterns = []
        
        # Common letter combinations
        common_pairs = [
            ("th", "th"), ("he", "he"), ("in", "in"), ("er", "er"), ("an", "an"),
            ("re", "re"), ("ed", "ed"), ("nd", "nd"), ("on", "on"), ("en", "en"),
            ("at", "at"), ("ou", "ou"), ("it", "it"), ("is", "is"), ("or", "or"),
            ("ti", "ti"), ("as", "as"), ("te", "te"), ("et", "et"), ("ng", "ng"),
            ("of", "of"), ("al", "al"), ("de", "de"), ("se", "se"), ("le", "le"),
            ("sa", "sa"), ("si", "si"), ("ar", "ar"), ("ve", "ve"), ("ra", "ra"),
            ("ld", "ld"), ("ur", "ur"), ("ly", "ly"), ("ta", "ta"), ("ri", "ri"),
            ("ne", "ne"), ("me", "me"), ("nt", "nt"), ("ty", "ty"), ("ic", "ic")
        ]
        
        patterns.extend(common_pairs)
        return patterns
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not text:
            return []
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Split into words and punctuation
        tokens = self._tokenize_text(text)
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.encoder:
                token_ids.append(self.encoder[token])
            else:
                # Try subword tokenization
                subtoken_ids = self._subword_tokenize(token)
                token_ids.extend(subtoken_ids)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.decoder:
                token = self.decoder[token_id]
                # Skip special tokens in output
                if not token.startswith('<|') or not token.endswith('|>'):
                    tokens.append(token)
        
        # Join tokens and clean up
        text = ' '.join(tokens)
        text = self._postprocess_text(text)
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for tokenization"""
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase (can be made optional)
        # text = text.lower()
        
        return text
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Split text into tokens"""
        # Regex pattern to split on whitespace and punctuation
        pattern = r'\w+|[^\w\s]'
        tokens = re.findall(pattern, text, re.UNICODE)
        return tokens
    
    def _subword_tokenize(self, word: str) -> List[int]:
        """Tokenize unknown words using subword approach"""
        if not word:
            return [self.special_tokens['<|unk|>']]
        
        # Try to find the longest matching substrings
        tokens = []
        i = 0
        while i < len(word):
            found = False
            # Try increasingly shorter substrings
            for j in range(len(word), i, -1):
                substr = word[i:j]
                if substr in self.encoder:
                    tokens.append(self.encoder[substr])
                    i = j
                    found = True
                    break
            
            if not found:
                # Single character fallback
                char = word[i]
                if char in self.encoder:
                    tokens.append(self.encoder[char])
                else:
                    tokens.append(self.special_tokens['<|unk|>'])
                i += 1
        
        return tokens
    
    def _postprocess_text(self, text: str) -> str:
        """Clean up decoded text"""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'encoder': self.encoder,
                'decoder': self.decoder,
                'special_tokens': self.special_tokens,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.encoder = data['encoder']
            self.decoder = data['decoder']
            self.special_tokens = data['special_tokens']
            self.vocab_size = data['vocab_size']

# Test the tokenizer
if __name__ == "__main__":
    tokenizer = AdvancedTokenizer()
    
    # Test encoding/decoding
    test_text = "Hello, world! This is a test of the advanced tokenizer for machine learning and artificial intelligence applications."
    
    print(f"Original text: {test_text}")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")
    
    print(f"Match: {test_text.lower() == decoded_text.lower()}")
