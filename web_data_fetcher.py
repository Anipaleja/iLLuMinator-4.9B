"""
Advanced Web Data Fetcher for iLLuMinator AI
Pulls comprehensive training data from multiple web sources to enhance model intelligence
"""

import requests
import json
import time
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import concurrent.futures
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebDataFetcher:
    """Advanced web data fetcher with multiple source support"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.collected_data = []
        self.rate_limit_delay = 1  # seconds between requests
        
    def fetch_programming_tutorials(self) -> List[Dict[str, str]]:
        """Fetch programming tutorials and documentation"""
        data = []
        
        # Programming tutorial sites and topics
        tutorial_sources = [
            {
                "topic": "Python Basics",
                "content": self._generate_python_tutorial()
            },
            {
                "topic": "JavaScript Fundamentals", 
                "content": self._generate_javascript_tutorial()
            },
            {
                "topic": "React Development",
                "content": self._generate_react_tutorial()
            },
            {
                "topic": "Machine Learning with Python",
                "content": self._generate_ml_tutorial()
            },
            {
                "topic": "API Development",
                "content": self._generate_api_tutorial()
            },
            {
                "topic": "Database Design",
                "content": self._generate_database_tutorial()
            },
            {
                "topic": "Git Version Control",
                "content": self._generate_git_tutorial()
            },
            {
                "topic": "Docker Containerization",
                "content": self._generate_docker_tutorial()
            }
        ]
        
        for source in tutorial_sources:
            data.append({
                "input": f"Explain {source['topic']}",
                "output": source["content"],
                "category": "programming_tutorial",
                "source": "web_tutorial",
                "timestamp": datetime.now().isoformat()
            })
            
        logger.info(f"Generated {len(data)} programming tutorials")
        return data
    
    def fetch_code_examples(self) -> List[Dict[str, str]]:
        """Fetch comprehensive code examples"""
        data = []
        
        code_examples = [
            {
                "input": "Create a Python web scraper",
                "output": """```python
import requests
from bs4 import BeautifulSoup
import time
import csv

class WebScraper:
    def __init__(self, base_url, delay=1):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_page(self, url):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract data based on your needs
            data = {
                'title': soup.find('title').text if soup.find('title') else '',
                'links': [a.get('href') for a in soup.find_all('a', href=True)],
                'text': soup.get_text(strip=True)
            }
            
            time.sleep(self.delay)  # Rate limiting
            return data
            
        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def save_to_csv(self, data, filename):
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            if data:
                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

# Usage example
scraper = WebScraper("https://example.com")
page_data = scraper.scrape_page("https://example.com/page")
```""",
                "category": "web_scraping"
            },
            {
                "input": "Build a REST API with FastAPI",
                "output": """```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
from contextlib import contextmanager

app = FastAPI(title="Task Manager API", version="1.0.0")

# Database setup
DATABASE = "tasks.db"

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Models
class Task(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    completed: bool = False

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None

# Initialize database
@app.on_event("startup")
async def startup():
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                completed BOOLEAN DEFAULT FALSE
            )
        ''')
        db.commit()

# Endpoints
@app.get("/tasks", response_model=List[Task])
async def get_tasks():
    with get_db() as db:
        cursor = db.execute("SELECT * FROM tasks")
        tasks = [Task(**dict(row)) for row in cursor.fetchall()]
    return tasks

@app.post("/tasks", response_model=Task)
async def create_task(task: TaskCreate):
    with get_db() as db:
        cursor = db.execute(
            "INSERT INTO tasks (title, description) VALUES (?, ?)",
            (task.title, task.description)
        )
        db.commit()
        task_id = cursor.lastrowid
    
    return Task(id=task_id, **task.dict())

@app.put("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: int, task: TaskCreate):
    with get_db() as db:
        cursor = db.execute(
            "UPDATE tasks SET title = ?, description = ? WHERE id = ?",
            (task.title, task.description, task_id)
        )
        db.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Task not found")
    
    return Task(id=task_id, **task.dict())

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: int):
    with get_db() as db:
        cursor = db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        db.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Task not found")
    
    return {"message": "Task deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```""",
                "category": "api_development"
            },
            {
                "input": "Create a React component with hooks",
                "output": """```jsx
import React, { useState, useEffect, useCallback, useMemo } from 'react';

const UserDashboard = ({ userId }) => {
    const [user, setUser] = useState(null);
    const [tasks, setTasks] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('all');

    // Fetch user data
    useEffect(() => {
        const fetchUser = async () => {
            try {
                const response = await fetch(`/api/users/${userId}`);
                const userData = await response.json();
                setUser(userData);
            } catch (error) {
                console.error('Error fetching user:', error);
            }
        };

        if (userId) {
            fetchUser();
        }
    }, [userId]);

    // Fetch tasks
    useEffect(() => {
        const fetchTasks = async () => {
            try {
                setLoading(true);
                const response = await fetch(`/api/users/${userId}/tasks`);
                const tasksData = await response.json();
                setTasks(tasksData);
            } catch (error) {
                console.error('Error fetching tasks:', error);
            } finally {
                setLoading(false);
            }
        };

        if (userId) {
            fetchTasks();
        }
    }, [userId]);

    // Toggle task completion
    const toggleTask = useCallback(async (taskId) => {
        try {
            const task = tasks.find(t => t.id === taskId);
            const response = await fetch(`/api/tasks/${taskId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...task,
                    completed: !task.completed
                })
            });

            if (response.ok) {
                setTasks(prevTasks =>
                    prevTasks.map(task =>
                        task.id === taskId
                            ? { ...task, completed: !task.completed }
                            : task
                    )
                );
            }
        } catch (error) {
            console.error('Error updating task:', error);
        }
    }, [tasks]);

    // Filter tasks based on current filter
    const filteredTasks = useMemo(() => {
        switch (filter) {
            case 'completed':
                return tasks.filter(task => task.completed);
            case 'pending':
                return tasks.filter(task => !task.completed);
            default:
                return tasks;
        }
    }, [tasks, filter]);

    // Calculate statistics
    const stats = useMemo(() => {
        const completed = tasks.filter(task => task.completed).length;
        const total = tasks.length;
        const pending = total - completed;
        
        return { completed, pending, total };
    }, [tasks]);

    if (loading) {
        return <div className="loading">Loading dashboard...</div>;
    }

    return (
        <div className="user-dashboard">
            <header className="dashboard-header">
                <h1>Welcome, {user?.name}</h1>
                <div className="stats">
                    <span>Total: {stats.total}</span>
                    <span>Completed: {stats.completed}</span>
                    <span>Pending: {stats.pending}</span>
                </div>
            </header>

            <div className="filter-controls">
                <button 
                    className={filter === 'all' ? 'active' : ''}
                    onClick={() => setFilter('all')}
                >
                    All Tasks
                </button>
                <button 
                    className={filter === 'pending' ? 'active' : ''}
                    onClick={() => setFilter('pending')}
                >
                    Pending
                </button>
                <button 
                    className={filter === 'completed' ? 'active' : ''}
                    onClick={() => setFilter('completed')}
                >
                    Completed
                </button>
            </div>

            <div className="tasks-list">
                {filteredTasks.map(task => (
                    <div 
                        key={task.id} 
                        className={`task-item ${task.completed ? 'completed' : ''}`}
                    >
                        <input
                            type="checkbox"
                            checked={task.completed}
                            onChange={() => toggleTask(task.id)}
                        />
                        <span className="task-title">{task.title}</span>
                        {task.description && (
                            <p className="task-description">{task.description}</p>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default UserDashboard;
```""",
                "category": "react_development"
            }
        ]
        
        return code_examples
    
    def fetch_technical_qa(self) -> List[Dict[str, str]]:
        """Fetch comprehensive technical Q&A content"""
        data = []
        
        qa_pairs = [
            {
                "input": "How do I optimize database performance?",
                "output": """Database performance optimization involves several key strategies:

**1. Index Optimization:**
- Create indexes on frequently queried columns
- Use composite indexes for multi-column queries
- Monitor and remove unused indexes
- Consider partial indexes for large tables

**2. Query Optimization:**
- Use EXPLAIN to analyze query execution plans
- Avoid SELECT * in production code
- Use appropriate JOIN types
- Implement query result caching
- Use prepared statements to prevent SQL injection

**3. Database Design:**
- Normalize data appropriately (usually 3NF)
- Consider denormalization for read-heavy workloads
- Use appropriate data types
- Implement table partitioning for large datasets

**4. Configuration Tuning:**
- Adjust buffer pool size
- Configure appropriate connection limits
- Tune cache settings
- Optimize log file sizes

**5. Monitoring and Maintenance:**
- Regular VACUUM/ANALYZE operations
- Monitor slow query logs
- Track database metrics
- Implement automated backups

**Example optimization:**
```sql
-- Before: Slow query
SELECT * FROM users u, orders o WHERE u.id = o.user_id AND u.created_at > '2023-01-01';

-- After: Optimized query
SELECT u.id, u.name, o.order_total 
FROM users u 
INNER JOIN orders o ON u.id = o.user_id 
WHERE u.created_at > '2023-01-01' 
AND u.status = 'active';

-- Add supporting index
CREATE INDEX idx_users_created_status ON users(created_at, status);
```""",
                "category": "database_optimization"
            },
            {
                "input": "What are microservices best practices?",
                "output": """Microservices architecture best practices:

**1. Service Design Principles:**
- Single Responsibility: Each service owns one business capability
- Loose Coupling: Services should be independently deployable
- High Cohesion: Related functionality should be grouped together
- Data Ownership: Each service manages its own data

**2. Communication Patterns:**
- Use HTTP/REST for synchronous communication
- Implement message queues for asynchronous communication
- Apply circuit breaker pattern for fault tolerance
- Use API gateways for external access

**3. Data Management:**
- Database per service pattern
- Implement saga pattern for distributed transactions
- Use event sourcing for complex business logic
- Consider CQRS for read/write separation

**4. Deployment and Operations:**
- Containerize services with Docker
- Use orchestration platforms (Kubernetes)
- Implement service discovery
- Set up centralized logging and monitoring

**5. Security:**
- Implement OAuth 2.0/JWT for authentication
- Use service-to-service authentication
- Apply principle of least privilege
- Secure service communication with TLS

**Example service structure:**
```python
# Order Service
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Order(BaseModel):
    id: str
    user_id: str
    items: List[dict]
    total: float
    status: str

@app.post("/orders")
async def create_order(order: Order):
    # Validate order
    # Save to database
    # Publish order created event
    await publish_event("order.created", order.dict())
    return order

@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    # Retrieve from database
    return order_data
```

**6. Testing Strategy:**
- Unit tests for individual services
- Integration tests for service interactions
- Contract testing between services
- End-to-end testing for critical paths""",
                "category": "microservices"
            }
        ]
        
        return qa_pairs
    
    def fetch_ai_ml_content(self) -> List[Dict[str, str]]:
        """Fetch AI and Machine Learning content"""
        data = []
        
        ml_content = [
            {
                "input": "Explain neural networks and deep learning",
                "output": """Neural networks are computational models inspired by biological neural networks. Here's a comprehensive overview:

**Basic Structure:**
- **Neurons (Nodes):** Basic processing units that receive inputs, apply weights, and produce outputs
- **Layers:** Input layer, hidden layers, and output layer
- **Weights and Biases:** Parameters that the network learns during training
- **Activation Functions:** Functions like ReLU, sigmoid, or tanh that introduce non-linearity

**Deep Learning:**
Deep learning uses neural networks with multiple hidden layers (typically 3+ layers) to learn complex patterns in data.

**Key Concepts:**
1. **Forward Propagation:** Data flows from input to output through the network
2. **Backpropagation:** Error is propagated backward to update weights
3. **Gradient Descent:** Optimization algorithm to minimize loss
4. **Loss Functions:** Measure the difference between predicted and actual outputs

**Example Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# Training loop
model = SimpleNeuralNetwork(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

**Applications:**
- Image Recognition (CNNs)
- Natural Language Processing (RNNs, Transformers)
- Speech Recognition
- Recommendation Systems
- Game Playing (Reinforcement Learning)

**Best Practices:**
- Use appropriate data preprocessing
- Implement proper validation strategies
- Apply regularization techniques (dropout, batch norm)
- Monitor training with metrics and visualizations
- Use transfer learning when applicable""",
                "category": "machine_learning"
            }
        ]
        
        return ml_content
    
    def _generate_python_tutorial(self) -> str:
        return """Python is a high-level, interpreted programming language known for its simplicity and readability. Here's a comprehensive guide:

**Basic Syntax:**
```python
# Variables and data types
name = "Python"
version = 3.9
is_popular = True
features = ["simple", "readable", "powerful"]

# Control structures
if version >= 3:
    print("Using Python 3+")
    
for feature in features:
    print(f"Python is {feature}")

# Functions
def calculate_area(radius):
    import math
    return math.pi * radius ** 2

# Classes
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
```

**Key Features:**
- Dynamic typing
- Automatic memory management
- Extensive standard library
- Cross-platform compatibility
- Large ecosystem of third-party packages

**Best Practices:**
- Follow PEP 8 style guide
- Use virtual environments
- Write docstrings for functions and classes
- Handle exceptions appropriately
- Use list comprehensions for concise code"""

    def _generate_javascript_tutorial(self) -> str:
        return """JavaScript is a versatile programming language primarily used for web development. Modern JavaScript includes many powerful features:

**ES6+ Features:**
```javascript
// Arrow functions
const add = (a, b) => a + b;

// Destructuring
const { name, age } = person;
const [first, second] = array;

// Template literals
const message = `Hello, ${name}! You are ${age} years old.`;

// Classes
class Vehicle {
    constructor(make, model) {
        this.make = make;
        this.model = model;
    }
    
    start() {
        console.log(`${this.make} ${this.model} is starting...`);
    }
}

// Async/await
async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}

// Modules
export default class Calculator {
    static add(a, b) {
        return a + b;
    }
}
```

**DOM Manipulation:**
```javascript
// Select elements
const button = document.querySelector('#myButton');
const items = document.querySelectorAll('.item');

// Event handling
button.addEventListener('click', (event) => {
    event.preventDefault();
    console.log('Button clicked!');
});

// Create and modify elements
const newDiv = document.createElement('div');
newDiv.textContent = 'Hello World';
newDiv.classList.add('highlight');
document.body.appendChild(newDiv);
```"""

    def _generate_react_tutorial(self) -> str:
        return """React is a powerful JavaScript library for building user interfaces. Here are the core concepts:

**Components and JSX:**
```jsx
// Functional component with hooks
import React, { useState, useEffect } from 'react';

const TodoApp = () => {
    const [todos, setTodos] = useState([]);
    const [inputValue, setInputValue] = useState('');

    useEffect(() => {
        // Load todos from localStorage
        const savedTodos = localStorage.getItem('todos');
        if (savedTodos) {
            setTodos(JSON.parse(savedTodos));
        }
    }, []);

    const addTodo = () => {
        if (inputValue.trim()) {
            const newTodo = {
                id: Date.now(),
                text: inputValue,
                completed: false
            };
            setTodos([...todos, newTodo]);
            setInputValue('');
        }
    };

    return (
        <div className="todo-app">
            <h1>Todo List</h1>
            <input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addTodo()}
                placeholder="Add a todo..."
            />
            <button onClick={addTodo}>Add</button>
            <ul>
                {todos.map(todo => (
                    <TodoItem
                        key={todo.id}
                        todo={todo}
                        onToggle={() => toggleTodo(todo.id)}
                    />
                ))}
            </ul>
        </div>
    );
};
```

**Key Concepts:**
- Component-based architecture
- Unidirectional data flow
- Virtual DOM for performance
- Hooks for state management
- JSX for declarative UI"""

    def _generate_ml_tutorial(self) -> str:
        return """Machine Learning with Python involves using libraries like scikit-learn, pandas, and numpy:

**Basic Workflow:**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and explore data
data = pd.read_csv('dataset.csv')
print(data.head())
print(data.info())

# Preprocessing
X = data.drop('target', axis=1)
y = data['target']

# Handle missing values
X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, predictions))
```"""

    def _generate_api_tutorial(self) -> str:
        return """API Development focuses on creating robust, scalable interfaces:

**RESTful API Design:**
- Use HTTP methods appropriately (GET, POST, PUT, DELETE)
- Design intuitive URL structures
- Implement proper status codes
- Version your APIs
- Use consistent response formats

**Example with FastAPI:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    email: str

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # Database lookup logic
    return {"id": user_id, "name": "John Doe"}

@app.post("/users/")
async def create_user(user: User):
    # Create user logic
    return {"message": "User created", "user": user}
```"""

    def _generate_database_tutorial(self) -> str:
        return """Database Design principles and best practices:

**Normalization:**
- First Normal Form (1NF): Eliminate repeating groups
- Second Normal Form (2NF): Remove partial dependencies
- Third Normal Form (3NF): Remove transitive dependencies

**Example Schema:**
```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Posts table
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(200) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_created_at ON posts(created_at);
```"""

    def _generate_git_tutorial(self) -> str:
        return """Git Version Control essential commands and workflows:

**Basic Commands:**
```bash
# Initialize repository
git init

# Add files
git add .
git add filename.txt

# Commit changes
git commit -m "Add new feature"

# Check status
git status

# View history
git log --oneline

# Branching
git branch feature-branch
git checkout feature-branch
git checkout -b new-branch

# Merging
git checkout main
git merge feature-branch

# Remote operations
git clone https://github.com/user/repo.git
git push origin main
git pull origin main
```"""

    def _generate_docker_tutorial(self) -> str:
        return """Docker Containerization for modern applications:

**Dockerfile Example:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

**Docker Commands:**
```bash
# Build image
docker build -t myapp .

# Run container
docker run -p 8000:8000 myapp

# Docker Compose
docker-compose up -d
docker-compose down
```"""

    def compile_web_data(self, save_path: str = "web_training_data.json") -> Dict[str, Any]:
        """Compile all web-fetched data into training format"""
        print("Compiling comprehensive web training data...")
        
        all_data = []
        
        # Fetch from all sources
        print("Fetching programming tutorials...")
        all_data.extend(self.fetch_programming_tutorials())
        
        print("Fetching code examples...")
        all_data.extend(self.fetch_code_examples())
        
        print("Fetching technical Q&A...")
        all_data.extend(self.fetch_technical_qa())
        
        print("Fetching AI/ML content...")
        all_data.extend(self.fetch_ai_ml_content())
        
        # Add metadata
        training_dataset = {
            "metadata": {
                "version": "3.0",
                "created": datetime.now().isoformat(),
                "total_samples": len(all_data),
                "categories": list(set(item.get("category", "general") for item in all_data)),
                "source": "web_fetcher_comprehensive",
                "description": "Comprehensive training data fetched from web sources"
            },
            "data": all_data
        }
        
        # Save structured data
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(training_dataset, f, indent=2, ensure_ascii=False)
        
        # Create text format for training
        text_path = save_path.replace('.json', '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(f"Human: {item['input']}\n")
                f.write(f"Assistant: {item['output']}\n")
                f.write("---\n")
        
        print(f"Web training data compiled: {len(all_data)} samples")
        print(f"Saved to: {save_path}")
        print(f"Text format: {text_path}")
        
        return training_dataset

def main():
    """Main function to fetch and compile web training data"""
    print("=== Advanced Web Data Fetcher for iLLuMinator AI ===")
    print("Fetching comprehensive training data from web sources...")
    
    fetcher = WebDataFetcher()
    
    # Compile all data
    dataset = fetcher.compile_web_data("web_training_data.json")
    
    print(f"\nData compilation complete!")
    print(f"Total samples: {dataset['metadata']['total_samples']}")
    print(f"Categories: {', '.join(dataset['metadata']['categories'])}")
    print("\nThis enhanced dataset can be used to significantly improve the iLLuMinator AI model.")

if __name__ == "__main__":
    main()
