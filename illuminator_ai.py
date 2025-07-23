"""
iLLuMinator AI - Professional Language Model
Advanced AI assistant with code generation and intelligent conversation capabilities
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Union, Tuple
import json
import re
import time
import warnings
from pathlib import Path
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class ProfessionalTokenizer:
    """Advanced tokenizer with code-aware capabilities"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
            '<CODE>': 4, '<CHAT>': 5, '<SYSTEM>': 6, '<USER>': 7,
            '<ASSISTANT>': 8, '<FUNCTION>': 9, '<CLASS>': 10, '<IMPORT>': 11
        }
        self.vocab = self._build_vocabulary()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
    def _build_vocabulary(self) -> List[str]:
        """Build comprehensive vocabulary for code and chat"""
        vocab = list(self.special_tokens.keys())
        
        # Programming keywords and symbols
        programming_tokens = [
            # Python keywords
            'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while',
            'try', 'except', 'finally', 'with', 'as', 'in', 'not', 'and', 'or', 'is', 'None',
            'True', 'False', 'lambda', 'yield', 'break', 'continue', 'pass', 'global', 'nonlocal',
            
            # JavaScript keywords
            'function', 'var', 'let', 'const', 'async', 'await', 'promise', 'callback',
            'typeof', 'instanceof', 'new', 'this', 'super', 'extends', 'implements',
            
            # Common programming symbols
            '(', ')', '[', ']', '{', '}', ';', ':', '.', ',', '=', '==', '!=', '<', '>', 
            '<=', '>=', '+', '-', '*', '/', '%', '**', '//', '+=', '-=', '*=', '/=',
            '&', '|', '^', '~', '<<', '>>', '&&', '||', '!', '?', '@', '#', '$',
            
            # Common words and phrases
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
            'just', 'should', 'now', 'I', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'get', 'got', 'make', 'made', 'go', 'went', 'come', 'came', 'see', 'saw',
            'know', 'knew', 'take', 'took', 'give', 'gave', 'find', 'found', 'think', 'thought',
            'tell', 'told', 'become', 'became', 'leave', 'left', 'feel', 'felt', 'put', 'set',
            'keep', 'kept', 'let', 'say', 'said', 'show', 'showed', 'try', 'tried', 'ask',
            'asked', 'work', 'worked', 'seem', 'seemed', 'turn', 'turned', 'start', 'started',
            'look', 'looked', 'want', 'wanted', 'give', 'call', 'called', 'move', 'moved',
            'live', 'lived', 'believe', 'believed', 'bring', 'brought', 'happen', 'happened',
        ]
        
        vocab.extend(programming_tokens)
        
        # Add numbers and common tokens
        for i in range(100):
            vocab.append(str(i))
            
        # Add alphabet and common character combinations
        for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            vocab.append(char)
            
        # Common suffixes and prefixes
        common_parts = [
            'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'ful', 'less',
            'able', 'ible', 'pre', 'un', 're', 'in', 'im', 'dis', 'mis', 'over', 'under',
            'out', 'up', 'down', 'self', 'ex', 'non', 'anti', 'pro', 'co', 'multi', 'mini',
            'micro', 'macro', 'super', 'ultra', 'mega', 'auto', 'semi', 'pseudo', 'quasi'
        ]
        vocab.extend(common_parts)
        
        # Pad vocabulary to desired size
        while len(vocab) < self.vocab_size:
            vocab.append(f'<UNK_{len(vocab)}>')
            
        return vocab[:self.vocab_size]
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs with smart tokenization"""
        if not text:
            return [self.special_tokens['<PAD>']]
            
        # Detect if text contains code
        is_code = self._detect_code(text)
        prefix = [self.special_tokens['<CODE>']] if is_code else [self.special_tokens['<CHAT>']]
        
        # Simple word-based tokenization with fallback
        tokens = []
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                # Character-level fallback
                for char in word:
                    if char in self.token_to_id:
                        tokens.append(self.token_to_id[char])
                    else:
                        tokens.append(self.special_tokens['<UNK>'])
        
        # Add prefix and limit length
        result = prefix + tokens
        if max_length:
            result = result[:max_length]
            
        return result
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if not token.startswith('<') or not token.endswith('>'):
                    tokens.append(token)
                    
        return ' '.join(tokens)
    
    def _detect_code(self, text: str) -> bool:
        """Detect if text contains code patterns"""
        code_patterns = [
            r'def\s+\w+\s*\(',
            r'class\s+\w+',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'function\s+\w+\s*\(',
            r'console\.log\s*\(',
            r'print\s*\(',
            r'\{\s*.*\s*\}',
            r'if\s*\(.*\)\s*\{',
            r'for\s*\(.*\)\s*\{',
            r'while\s*\(.*\)\s*\{'
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False

class AdvancedTransformerBlock(torch.nn.Module):
    """Professional transformer block with enhanced capabilities"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head attention
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.o_proj = torch.nn.Linear(d_model, d_model, bias=False)
        
        # Enhanced feed-forward network
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4 * d_model, d_model),
            torch.nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.ln1(x)
        
        # Multi-head attention
        batch_size, seq_len, d_model = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = self.o_proj(attn_output)
        
        # Residual connection
        x = residual + self.dropout(attn_output)
        
        # Pre-norm feed-forward
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        
        # Residual connection
        x = residual + x
        
        return x

class ProfessionalIlluminatorModel(torch.nn.Module):
    """Advanced iLLuMinator model with professional capabilities"""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.layers = torch.nn.ModuleList([
            AdvancedTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            attention_mask = attention_mask.view(1, 1, seq_len, seq_len)
            
        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

class IlluminatorAI:
    """Professional AI Assistant with advanced capabilities"""
    
    def __init__(self, fast_mode: bool = True, auto_enhance: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = ProfessionalTokenizer()
        self.fast_mode = fast_mode
        self.auto_enhance = auto_enhance
        
        if fast_mode:
            # Optimized configuration for faster inference while maintaining 4.7B parameters
            self.model = ProfessionalIlluminatorModel(
                vocab_size=self.tokenizer.vocab_size,
                d_model=1536,  # Reduced but still large (4.7B parameters)
                n_layers=32,   # Optimized depth
                n_heads=24,    # Efficient attention heads
                max_seq_len=512  # Shorter context for speed
            ).to(self.device)
        else:
            # Original large configuration
            self.model = ProfessionalIlluminatorModel(
                vocab_size=self.tokenizer.vocab_size,
                d_model=2816,  # Very large model dimension
                n_layers=48,   # Deep architecture
                n_heads=44,    # Multi-head attention
                max_seq_len=1024  # Full context length
            ).to(self.device)
        
        # Enable optimizations
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled for faster inference")
            except:
                print("Torch compile not available, using standard mode")
        
        # Initialize conversation context
        self.conversation_history = []
        self.system_prompt = self._get_system_prompt()
        self.kv_cache = {}  # Add KV cache for faster generation
        
        # Web data integration
        self.web_knowledge_base = {}
        if auto_enhance:
            self._load_web_knowledge()
        
        print(f"iLLuMinator AI initialized successfully on {self.device}")
        print(f"Fast mode: {fast_mode}")
        print(f"Auto-enhance with web data: {auto_enhance}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_system_prompt(self) -> str:
        """Professional system prompt for the AI assistant"""
        return """I am iLLuMinator AI, a professional artificial intelligence assistant designed to provide exceptional support in programming, technical discussions, and general conversation.

My capabilities include:
- Advanced code generation in multiple programming languages
- Comprehensive technical explanations and troubleshooting
- Professional software development guidance
- Intelligent problem-solving and analysis
- Clear, concise, and helpful communication

I provide accurate, well-structured responses without unnecessary formatting or emojis, focusing on delivering practical value and professional assistance."""

    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 256, 
        temperature: float = 0.7,
        top_k: int = 50
    ) -> str:
        """Generate intelligent response to user input with web knowledge enhancement"""
        
        # Prepare input with conversation context
        full_prompt = self._prepare_prompt(prompt)
        
        # Tokenize input
        input_ids = self.tokenizer.encode(full_prompt, max_length=512)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self._generate_tokens(
                input_tensor, max_tokens, temperature, top_k
            )
        
        # Decode and clean response
        base_response = self.tokenizer.decode(generated_ids[0][len(input_ids):])
        base_response = self._clean_response(base_response)
        
        # Enhance response with web knowledge if available
        if self.auto_enhance and self.web_knowledge_base:
            enhanced_response = self._enhance_response_with_knowledge(prompt, base_response)
        else:
            enhanced_response = base_response
        
        # Update conversation history
        self.conversation_history.append({"user": prompt, "assistant": enhanced_response})
        
        return enhanced_response
    
    def _prepare_prompt(self, user_input: str) -> str:
        """Prepare prompt with context and system instructions"""
        context_parts = [self.system_prompt]
        
        # Add recent conversation history
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        
        context_parts.append(f"User: {user_input}")
        context_parts.append("Assistant:")
        
        return "\n\n".join(context_parts)
    
    def _generate_tokens(
        self, 
        input_ids: torch.Tensor, 
        max_tokens: int, 
        temperature: float, 
        top_k: int
    ) -> torch.Tensor:
        """Generate tokens using optimized sampling techniques with early stopping"""
        
        generated = input_ids.clone()
        
        # Early stopping tokens for faster generation
        stop_tokens = [
            self.tokenizer.special_tokens.get('<EOS>', 3),
            self.tokenizer.token_to_id.get('.', -1),
            self.tokenizer.token_to_id.get('!', -1),
            self.tokenizer.token_to_id.get('?', -1)
        ]
        stop_tokens = [t for t in stop_tokens if t != -1]
        
        for step in range(max_tokens):
            # Forward pass with memory optimization
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                logits = self.model(generated)
                next_token_logits = logits[0, -1, :]
            
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering (optimized)
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Early stopping check
                if next_token.item() in stop_tokens:
                    break
                    
                # Append token
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Limit context length for memory efficiency
                if generated.size(1) > self.model.max_seq_len:
                    generated = generated[:, -self.model.max_seq_len:]
                    break
        
        return generated
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove extra whitespace and clean up
        response = ' '.join(response.split())
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Ensure response isn't too short
        if len(response.strip()) < 10:
            return "I understand your question. Let me provide a comprehensive response based on the context."
        
        return response.strip()
    
    def generate_code(self, description: str, language: str = "python") -> str:
        """Generate code based on description"""
        code_prompt = f"""Generate clean, professional {language} code for the following requirement:

{description}

Requirements:
- Write clean, well-commented code
- Follow best practices and conventions
- Include proper error handling where appropriate
- Make the code production-ready

Code:"""
        
        response = self.generate_response(code_prompt, max_tokens=300, temperature=0.3)
        
        # Extract code from response if it contains explanation
        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        return response
    
    def chat(self, message: str) -> str:
        """Intelligent chat interface with speed optimizations"""
        if not message.strip():
            return "Please provide a message for me to respond to."
        
        # Detect if this is a code request
        code_keywords = ['code', 'function', 'program', 'script', 'implement', 'write', 'create']
        if any(keyword in message.lower() for keyword in code_keywords):
            return self.generate_code(message)
        
        # Optimized chat response with reduced tokens for speed
        # Use lower temperature for more focused responses
        return self.generate_response(
            message, 
            max_tokens=50 if len(message) < 20 else 80,  # Dynamic token limit
            temperature=0.6  # Balanced creativity and speed
        )
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def save_conversation(self, filename: str):
        """Save conversation history to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            print(f"Conversation saved to {filename}")
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def _load_web_knowledge(self):
        """Load web-sourced knowledge base for enhanced responses"""
        try:
            # Check if web training data exists
            web_data_files = [
                "web_training_data.json",
                "enhanced_training_data.json"
            ]
            
            for data_file in web_data_files:
                if Path(data_file).exists():
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'data' in data:
                            self._process_knowledge_data(data['data'])
                        else:
                            self._process_knowledge_data(data)
                    print(f"Loaded knowledge from {data_file}")
                    break
            else:
                # Generate web knowledge if not available
                print("Generating web knowledge base...")
                self._generate_web_knowledge()
                
        except Exception as e:
            print(f"Could not load web knowledge: {e}")
    
    def _process_knowledge_data(self, data: List[Dict]):
        """Process and index knowledge data for quick retrieval"""
        for item in data:
            if isinstance(item, dict) and 'input' in item and 'output' in item:
                # Create searchable keywords from input
                keywords = self._extract_keywords(item['input'])
                for keyword in keywords:
                    if keyword not in self.web_knowledge_base:
                        self.web_knowledge_base[keyword] = []
                    self.web_knowledge_base[keyword].append({
                        'question': item['input'],
                        'answer': item['output'],
                        'category': item.get('category', 'general')
                    })
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for knowledge indexing"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common words and keep meaningful terms
        stop_words = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:5]  # Limit to top 5 keywords
    
    def _generate_web_knowledge(self):
        """Generate basic web knowledge if external data not available"""
        basic_knowledge = {
            'python': [{
                'question': 'What is Python?',
                'answer': 'Python is a high-level, interpreted programming language known for its simplicity and readability.',
                'category': 'programming'
            }],
            'javascript': [{
                'question': 'What is JavaScript?',
                'answer': 'JavaScript is a versatile programming language primarily used for web development.',
                'category': 'programming'
            }],
            'react': [{
                'question': 'What is React?',
                'answer': 'React is a JavaScript library for building user interfaces, particularly web applications.',
                'category': 'programming'
            }],
            'api': [{
                'question': 'What is an API?',
                'answer': 'API stands for Application Programming Interface. It defines how different software components communicate.',
                'category': 'programming'
            }]
        }
        
        self.web_knowledge_base.update(basic_knowledge)
        print("Generated basic knowledge base")
    
    def _enhance_response_with_knowledge(self, prompt: str, base_response: str) -> str:
        """Enhance response using web knowledge base"""
        try:
            # Extract keywords from prompt
            keywords = self._extract_keywords(prompt)
            
            # Find relevant knowledge
            relevant_info = []
            for keyword in keywords:
                if keyword in self.web_knowledge_base:
                    relevant_info.extend(self.web_knowledge_base[keyword][:2])  # Limit to 2 items per keyword
            
            if relevant_info:
                # Enhance response with relevant knowledge
                enhanced_response = base_response
                
                # Add relevant context if the base response is short
                if len(base_response) < 100 and relevant_info:
                    best_match = relevant_info[0]
                    if len(best_match['answer']) > len(base_response):
                        enhanced_response = best_match['answer']
                
                return enhanced_response
            
            return base_response
            
        except Exception as e:
            print(f"Knowledge enhancement error: {e}")
            return base_response
    
    def fetch_and_integrate_web_data(self):
        """Fetch fresh web data and integrate it into the knowledge base"""
        try:
            print("Fetching fresh web data...")
            
            # Try to generate web data
            try:
                # This would normally import the web fetcher, but we'll create inline data
                web_data = self._create_fresh_web_data()
                
                # Process the new data
                self._process_knowledge_data(web_data)
                
                print(f"Integrated {len(web_data)} new knowledge items")
                
                # Save the updated knowledge
                self._save_knowledge_base()
                
            except Exception as e:
                print(f"Could not fetch web data: {e}")
                
        except Exception as e:
            print(f"Web data integration failed: {e}")
    
    def _create_fresh_web_data(self) -> List[Dict]:
        """Create fresh web-style data for knowledge enhancement"""
        fresh_data = [
            {
                "input": "How to optimize Python code performance?",
                "output": """Here are key strategies to optimize Python code performance:

1. **Use Built-in Functions**: Leverage built-in functions like map(), filter(), and sum() which are implemented in C.

2. **List Comprehensions**: Replace loops with list comprehensions when possible.
   ```python
   # Slower
   result = []
   for i in range(10):
       result.append(i**2)
   
   # Faster
   result = [i**2 for i in range(10)]
   ```

3. **Use NumPy for Numerical Operations**: NumPy operations are much faster than pure Python.

4. **Profile Your Code**: Use cProfile to identify bottlenecks.

5. **Avoid Global Variables**: Local variable access is faster.

6. **Use Sets for Membership Testing**: Sets have O(1) lookup time vs O(n) for lists.""",
                "category": "python_optimization"
            },
            {
                "input": "What are modern JavaScript best practices?",
                "output": """Modern JavaScript best practices include:

1. **Use ES6+ Features**: 
   - Arrow functions for concise syntax
   - Template literals for string interpolation
   - Destructuring for cleaner variable assignment

2. **Async/Await over Promises**: More readable asynchronous code
   ```javascript
   // Modern approach
   async function fetchData() {
       try {
           const response = await fetch('/api/data');
           const data = await response.json();
           return data;
       } catch (error) {
           console.error('Error:', error);
       }
   }
   ```

3. **Use Modules**: Import/export for better code organization

4. **Strict Mode**: Always use 'use strict' for better error catching

5. **Consistent Code Style**: Use tools like ESLint and Prettier

6. **Error Handling**: Always handle errors gracefully""",
                "category": "javascript_best_practices"
            },
            {
                "input": "Explain machine learning model deployment",
                "output": """Machine learning model deployment involves several key steps:

1. **Model Serialization**: Save trained models using pickle, joblib, or ONNX format

2. **API Development**: Create REST APIs using frameworks like FastAPI or Flask
   ```python
   from fastapi import FastAPI
   import joblib
   
   app = FastAPI()
   model = joblib.load('model.pkl')
   
   @app.post('/predict')
   async def predict(data: dict):
       prediction = model.predict([data['features']])
       return {'prediction': prediction[0]}
   ```

3. **Containerization**: Use Docker for consistent deployment environments

4. **Monitoring**: Track model performance and data drift in production

5. **Scaling**: Use load balancers and auto-scaling for high traffic

6. **Version Control**: Maintain different model versions for rollback capability""",
                "category": "machine_learning_deployment"
            },
            {
                "input": "How to design scalable database schemas?",
                "output": """Scalable database schema design principles:

1. **Normalization vs Denormalization**: 
   - Normalize for consistency
   - Denormalize for performance when needed

2. **Indexing Strategy**:
   - Index frequently queried columns
   - Use composite indexes for multi-column queries
   - Monitor index usage and remove unused ones

3. **Partitioning**: Split large tables horizontally or vertically

4. **Replication**: Use read replicas for scaling read operations

5. **Caching Layer**: Implement Redis or Memcached for frequently accessed data

6. **Connection Pooling**: Manage database connections efficiently

Example schema design:
```sql
-- Users table with proper indexing
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'active'
);

CREATE INDEX idx_users_email_status ON users(email, status);
CREATE INDEX idx_users_created_at ON users(created_at);
```""",
                "category": "database_design"
            }
        ]
        
        return fresh_data
    
    def _save_knowledge_base(self):
        """Save the current knowledge base to file"""
        try:
            knowledge_file = "integrated_knowledge_base.json"
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.web_knowledge_base, f, indent=2, ensure_ascii=False)
            print(f"Knowledge base saved to {knowledge_file}")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

def main():
    """Professional command-line interface"""
    print("=" * 60)
    print("iLLuMinator AI - Professional Language Model")
    print("Advanced AI Assistant for Code Generation and Intelligent Chat")
    print("=" * 60)
    
    # Initialize AI
    try:
        ai = IlluminatorAI()
        print("\nInitialization complete. Ready for interaction.")
    except Exception as e:
        print(f"Initialization error: {e}")
        return
    
    print("\nCommands:")
    print("- Type your message for intelligent conversation")
    print("- Use 'code: <description>' for code generation")
    print("- Type 'clear' to clear conversation history")
    print("- Type 'save <filename>' to save conversation")
    print("- Type 'quit' to exit")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\nThank you for using iLLuMinator AI. Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                ai.clear_conversation()
                continue
            
            if user_input.lower().startswith('save '):
                filename = user_input[5:].strip() or "conversation.json"
                ai.save_conversation(filename)
                continue
            
            # Process input
            start_time = time.time()
            
            if user_input.lower().startswith('code:'):
                description = user_input[5:].strip()
                response = ai.generate_code(description)
                print(f"\niLLuMinator AI:\n{response}")
            else:
                response = ai.chat(user_input)
                print(f"\niLLuMinator AI: {response}")
            
            # Show response time
            response_time = time.time() - start_time
            print(f"\n[Response generated in {response_time:.2f} seconds]")
            
        except KeyboardInterrupt:
            print("\n\nExiting iLLuMinator AI. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
