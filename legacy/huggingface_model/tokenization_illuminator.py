"""
Enhanced Tokenizer for Hugging Face Integration
Improved accuracy with comprehensive vocabulary and encoding
"""

import json
import re
from typing import List, Dict, Optional, Union
from transformers import PreTrainedTokenizer
import os

class IlluminatorTokenizer(PreTrainedTokenizer):
    """
    Enhanced tokenizer for the Illuminator model with improved accuracy
    Compatible with Hugging Face transformers
    """
    
    vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        errors="replace",
        unk_token="<|unk|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        add_prefix_space=False,
        **kwargs
    ):
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs
        )
        
        self.add_prefix_space = add_prefix_space
        
        # Initialize enhanced vocabulary
        if vocab_file and os.path.isfile(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.encoder = json.load(f)
        else:
            self.encoder = self._build_enhanced_vocabulary()
        
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Enhanced BPE merges for better subword handling
        self.bpe_merges = []
        if merges_file and os.path.isfile(merges_file):
            with open(merges_file, 'r', encoding='utf-8') as f:
                self.bpe_merges = [tuple(line.strip().split()) for line in f.readlines()[1:]]
        else:
            self.bpe_merges = self._build_enhanced_bpe_merges()
        
        self.bpe_merges_dict = dict(self.bpe_merges)
        self.cache = {}
    
    def _build_enhanced_vocabulary(self) -> Dict[str, int]:
        """Build comprehensive vocabulary for maximum accuracy"""
        vocab = {}
        idx = 0
        
        # Special tokens first
        special_tokens = [
            "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", 
            "<|mask|>", "<|sep|>", "<|cls|>", "<|endoftext|>"
        ]
        
        for token in special_tokens:
            vocab[token] = idx
            idx += 1
        
        # Bytes for all possible byte values (0-255)
        for i in range(256):
            vocab[chr(i)] = idx
            idx += 1
        
        # Enhanced vocabulary for better accuracy
        enhanced_words = self._get_enhanced_vocabulary_words()
        for word in enhanced_words:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
        
        # Common subwords and morphemes
        subwords = self._get_subword_vocabulary()
        for subword in subwords:
            if subword not in vocab:
                vocab[subword] = idx
                idx += 1
        
        # Technical terms for better domain coverage
        technical_terms = self._get_technical_vocabulary()
        for term in technical_terms:
            if term not in vocab:
                vocab[term] = idx
                idx += 1
        
        return vocab
    
    def _get_enhanced_vocabulary_words(self) -> List[str]:
        """Get enhanced vocabulary for better accuracy"""
        return [
            # High-frequency words
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
            
            # AI/ML terms for domain accuracy
            "artificial", "intelligence", "machine", "learning", "deep", "neural", "network", "algorithm", "model", "training", "data", "dataset",
            "feature", "prediction", "classification", "regression", "supervised", "unsupervised", "reinforcement", "attention", "transformer",
            "embedding", "gradient", "optimization", "backpropagation", "epoch", "batch", "loss", "accuracy", "validation", "testing",
            
            # Programming terms
            "python", "javascript", "java", "cpp", "function", "method", "class", "object", "variable", "parameter", "return", "loop",
            "condition", "array", "list", "dictionary", "string", "integer", "boolean", "algorithm", "structure", "framework", "library",
            
            # Science terms
            "physics", "chemistry", "biology", "mathematics", "quantum", "relativity", "evolution", "genetics", "climate", "environment",
            "energy", "force", "matter", "atom", "molecule", "cell", "organism", "ecosystem", "theory", "experiment", "research",
            
            # Technology terms
            "computer", "software", "hardware", "internet", "network", "database", "security", "encryption", "server", "client",
            "protocol", "application", "system", "platform", "technology", "digital", "electronic", "innovation", "development",
            
            # Common prefixes and suffixes
            "un", "re", "in", "dis", "en", "non", "over", "mis", "sub", "pre", "inter", "fore", "de", "trans", "super", "semi", "anti",
            "ing", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment", "ful", "less", "able", "ible", "ous", "ious", "ive",
        ]
    
    def _get_subword_vocabulary(self) -> List[str]:
        """Get subword vocabulary for better tokenization"""
        return [
            # Common letter combinations
            "th", "he", "in", "er", "an", "re", "ed", "nd", "on", "en", "at", "ou", "it", "is", "or", "ti", "as", "te", "et", "ng",
            "of", "al", "de", "se", "le", "sa", "si", "ar", "ve", "ra", "ld", "ur", "ly", "ta", "ri", "ne", "me", "nt", "ty", "ic",
            
            # Programming patterns
            "def", "class", "import", "from", "return", "if", "else", "elif", "for", "while", "try", "except", "with", "lambda",
            "self", "init", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple", "range", "print", "input",
            
            # Technical patterns
            "http", "https", "www", "com", "org", "net", "api", "json", "xml", "html", "css", "sql", "url", "uri", "uuid",
            "config", "setup", "install", "version", "update", "upgrade", "debug", "error", "warning", "info", "log",
        ]
    
    def _get_technical_vocabulary(self) -> List[str]:
        """Get technical vocabulary for domain expertise"""
        return [
            # AI/ML frameworks and tools
            "pytorch", "tensorflow", "keras", "scikit", "pandas", "numpy", "matplotlib", "jupyter", "colab", "huggingface",
            "openai", "anthropic", "deepmind", "nvidia", "cuda", "gpu", "cpu", "ram", "memory", "storage",
            
            # Cloud and infrastructure
            "aws", "azure", "gcp", "docker", "kubernetes", "linux", "ubuntu", "centos", "debian", "windows",
            "server", "cluster", "container", "virtual", "machine", "instance", "deployment", "scaling",
            
            # Programming languages and frameworks
            "react", "angular", "vue", "nodejs", "express", "django", "flask", "fastapi", "spring", "laravel",
            "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "kafka", "rabbitmq", "nginx",
            
            # Version control and development
            "git", "github", "gitlab", "bitbucket", "branch", "commit", "merge", "pull", "push", "clone",
            "repository", "fork", "issue", "release", "tag", "workflow", "pipeline", "cicd", "devops",
        ]
    
    def _build_enhanced_bpe_merges(self) -> List[tuple]:
        """Build enhanced BPE merges for better subword tokenization"""
        return [
            # Common English patterns
            ("t", "h"), ("h", "e"), ("i", "n"), ("e", "r"), ("a", "n"), ("r", "e"), ("e", "d"), ("n", "d"),
            ("o", "n"), ("e", "n"), ("a", "t"), ("o", "u"), ("i", "t"), ("i", "s"), ("o", "r"), ("t", "i"),
            ("a", "s"), ("t", "e"), ("e", "t"), ("n", "g"), ("o", "f"), ("a", "l"), ("d", "e"), ("s", "e"),
            
            # Programming patterns
            ("d", "ef"), ("cl", "ass"), ("im", "port"), ("fr", "om"), ("ret", "urn"), ("sel", "f"),
            ("in", "it"), ("le", "n"), ("st", "r"), ("in", "t"), ("pr", "int"), ("ran", "ge"),
            
            # Technical patterns
            ("ht", "tp"), ("ww", "w"), ("co", "m"), ("or", "g"), ("ne", "t"), ("ap", "i"),
            ("js", "on"), ("ht", "ml"), ("cs", "s"), ("sq", "l"), ("ur", "l"), ("uu", "id"),
            
            # AI/ML patterns
            ("ne", "ural"), ("net", "work"), ("mod", "el"), ("tra", "in"), ("dat", "a"), ("acc", "uracy"),
            ("los", "s"), ("gra", "dient"), ("opt", "im"), ("bat", "ch"), ("epo", "ch"), ("val", "id"),
        ]
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary dictionary"""
        return self.encoder.copy()
    
    @property
    def vocab_size(self) -> int:
        """Return the size of vocabulary"""
        return len(self.encoder)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using enhanced BPE"""
        if not text:
            return []
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Split into words
        words = re.findall(r'\S+|\s+', text)
        
        tokens = []
        for word in words:
            if word.isspace():
                continue
            
            # Apply BPE to each word
            word_tokens = self._bpe_encode(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better tokenization"""
        # Handle Unicode normalization
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        
        # Handle common programming patterns
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase -> camel Case
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # word123 -> word 123
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # 123word -> 123 word
        
        return text
    
    def _bpe_encode(self, word: str) -> List[str]:
        """Apply BPE encoding to a word"""
        if word in self.cache:
            return self.cache[word]
        
        # Convert to list of characters
        word_chars = list(word)
        
        if len(word_chars) == 1:
            return word_chars
        
        # Apply BPE merges
        while len(word_chars) > 1:
            pairs = self._get_pairs(word_chars)
            if not pairs:
                break
            
            # Find the best pair to merge
            best_pair = min(pairs, key=lambda x: self.bpe_merges_dict.get(x, float('inf')))
            
            if best_pair not in self.bpe_merges_dict:
                break
            
            # Merge the best pair
            new_word_chars = []
            i = 0
            while i < len(word_chars):
                if (i < len(word_chars) - 1 and 
                    word_chars[i] == best_pair[0] and 
                    word_chars[i + 1] == best_pair[1]):
                    new_word_chars.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word_chars.append(word_chars[i])
                    i += 1
            
            word_chars = new_word_chars
        
        self.cache[word] = word_chars
        return word_chars
    
    def _get_pairs(self, word_chars: List[str]) -> set:
        """Get all adjacent pairs in the word"""
        pairs = set()
        for i in range(len(word_chars) - 1):
            pairs.add((word_chars[i], word_chars[i + 1]))
        return pairs
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID"""
        return self.encoder.get(token, self.encoder.get(self.unk_token))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token"""
        return self.decoder.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string"""
        text = ''.join(tokens)
        
        # Clean up the text
        text = text.replace('</w>', ' ')
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Save vocabulary files"""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        merges_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "merges.txt"
        )
        
        # Save vocabulary
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, indent=2, sort_keys=True, ensure_ascii=False)
        
        # Save merges
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write('#version: 0.2\n')
            for merge in self.bpe_merges:
                f.write(f'{merge[0]} {merge[1]}\n')
        
        return vocab_file, merges_file
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """Build model inputs by adding special tokens"""
        bos = [self.bos_token_id] if self.bos_token_id is not None else []
        eos = [self.eos_token_id] if self.eos_token_id is not None else []
        
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        
        sep = [self.sep_token_id] if hasattr(self, 'sep_token_id') and self.sep_token_id is not None else []
        return bos + token_ids_0 + sep + token_ids_1 + eos
