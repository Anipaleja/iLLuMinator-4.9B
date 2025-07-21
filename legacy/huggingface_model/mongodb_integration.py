"""
Advanced MongoDB Integration for Quantum Illuminator
Enterprise-grade database operations with vector indexing and real-time analytics
"""

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import gridfs
from bson import ObjectId

class QueryType(Enum):
    """Query classification for analytics and optimization"""
    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    CONVERSATIONAL = "conversational"
    CODE_GENERATION = "code_generation"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    QUANTUM_ENHANCED = "quantum_enhanced"

@dataclass
class ConversationRecord:
    """Structured conversation record for MongoDB storage"""
    conversation_id: str
    user_query: str
    model_response: str
    query_type: QueryType
    confidence_score: float
    processing_time_ms: int
    quantum_enhancement_used: bool
    vector_embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    model_version: str
    performance_metrics: Dict[str, float]

@dataclass
class ModelPerformanceMetrics:
    """Comprehensive performance tracking"""
    model_version: str
    total_queries: int
    average_response_time: float
    confidence_distribution: Dict[str, int]
    query_type_distribution: Dict[str, int]
    quantum_enhancement_usage: float
    error_rate: float
    user_satisfaction_score: float
    timestamp: datetime

class AdvancedMongoManager:
    """
    Enterprise-grade MongoDB integration with advanced features:
    - Vector similarity search
    - Real-time analytics
    - Performance monitoring
    - Conversation history management
    - Quantum state persistence
    """
    
    def __init__(
        self, 
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "quantum_illuminator_db",
        enable_async: bool = True
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.enable_async = enable_async
        
        # Initialize connections
        self.client = MongoClient(connection_string)
        self.db: Database = self.client[database_name]
        
        if enable_async:
            self.async_client = AsyncIOMotorClient(connection_string)
            self.async_db = self.async_client[database_name]
        
        # Initialize GridFS for large data storage
        self.fs = gridfs.GridFS(self.db)
        
        # Initialize collections
        self.conversations: Collection = self.db.conversations
        self.performance_metrics: Collection = self.db.performance_metrics
        self.quantum_states: Collection = self.db.quantum_states
        self.vector_embeddings: Collection = self.db.vector_embeddings
        self.user_feedback: Collection = self.db.user_feedback
        self.model_versions: Collection = self.db.model_versions
        self.analytics_cache: Collection = self.db.analytics_cache
        
        # Setup indexes for optimal performance
        self._setup_indexes()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"MongoDB Manager initialized for database: {database_name}")
    
    def _setup_indexes(self):
        """Create optimized indexes for high-performance queries"""
        
        # Conversation indexes
        self.conversations.create_index([
            ("timestamp", -1),
            ("query_type", 1),
            ("confidence_score", -1)
        ])
        self.conversations.create_index("conversation_id", unique=True)
        self.conversations.create_index([("user_query", "text"), ("model_response", "text")])
        
        # Vector similarity index
        self.vector_embeddings.create_index([
            ("embedding", "2dsphere")  # For vector similarity search
        ])
        self.vector_embeddings.create_index([
            ("query_hash", 1),
            ("timestamp", -1)
        ])
        
        # Performance metrics indexes
        self.performance_metrics.create_index([
            ("timestamp", -1),
            ("model_version", 1)
        ])
        
        # Quantum states indexes
        self.quantum_states.create_index([
            ("state_hash", 1),
            ("coherence_time", -1)
        ])
        
        self.logger.info("Database indexes created successfully")
    
    def store_conversation(
        self, 
        conversation_record: ConversationRecord
    ) -> str:
        """
        Store conversation with comprehensive metadata and vector embeddings
        
        Args:
            conversation_record: Structured conversation data
            
        Returns:
            MongoDB document ID
        """
        try:
            # Convert to dictionary for MongoDB storage
            record_dict = asdict(conversation_record)
            
            # Generate query hash for deduplication
            query_hash = hashlib.sha256(
                conversation_record.user_query.encode('utf-8')
            ).hexdigest()
            record_dict['query_hash'] = query_hash
            
            # Store conversation
            result = self.conversations.insert_one(record_dict)
            
            # Store vector embedding separately for efficient similarity search
            embedding_doc = {
                "conversation_id": conversation_record.conversation_id,
                "query_hash": query_hash,
                "embedding": conversation_record.vector_embedding,
                "query_type": conversation_record.query_type.value,
                "timestamp": conversation_record.timestamp
            }
            self.vector_embeddings.insert_one(embedding_doc)
            
            self.logger.info(f"Conversation stored: {conversation_record.conversation_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Error storing conversation: {str(e)}")
            raise
    
    def find_similar_conversations(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find conversations with similar vector embeddings using advanced similarity search
        
        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar conversation records
        """
        try:
            # Convert embedding to numpy array for computation
            query_vector = np.array(query_embedding)
            
            # Get candidate embeddings (in production, use proper vector database)
            candidates = list(self.vector_embeddings.find({}, {
                "conversation_id": 1,
                "embedding": 1,
                "query_type": 1,
                "timestamp": 1
            }).limit(1000))  # Limit for performance
            
            similarities = []
            for candidate in candidates:
                candidate_vector = np.array(candidate["embedding"])
                
                # Cosine similarity
                similarity = np.dot(query_vector, candidate_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(candidate_vector)
                )
                
                if similarity >= similarity_threshold:
                    similarities.append({
                        "conversation_id": candidate["conversation_id"],
                        "similarity_score": float(similarity),
                        "query_type": candidate["query_type"],
                        "timestamp": candidate["timestamp"]
                    })
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Get full conversation details for top matches
            similar_conversations = []
            for sim in similarities[:limit]:
                conversation = self.conversations.find_one({
                    "conversation_id": sim["conversation_id"]
                })
                if conversation:
                    conversation["similarity_score"] = sim["similarity_score"]
                    similar_conversations.append(conversation)
            
            return similar_conversations
            
        except Exception as e:
            self.logger.error(f"Error finding similar conversations: {str(e)}")
            return []
    
    def store_quantum_state(
        self, 
        quantum_state: Dict[str, Any],
        coherence_time: int,
        entanglement_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store quantum state information for advanced quantum computing research
        
        Args:
            quantum_state: Quantum state parameters and measurements
            coherence_time: How long the quantum state remains coherent
            entanglement_data: Quantum entanglement information
            
        Returns:
            MongoDB document ID
        """
        try:
            # Generate state hash for uniqueness
            state_hash = hashlib.sha256(
                json.dumps(quantum_state, sort_keys=True).encode()
            ).hexdigest()
            
            quantum_record = {
                "state_hash": state_hash,
                "quantum_state": quantum_state,
                "coherence_time": coherence_time,
                "entanglement_data": entanglement_data or {},
                "timestamp": datetime.now(timezone.utc),
                "measurements": {
                    "fidelity": np.random.uniform(0.8, 0.99),  # Simulated quantum fidelity
                    "purity": np.random.uniform(0.7, 0.95),   # Simulated quantum purity
                    "entanglement_entropy": np.random.uniform(0.1, 0.8)
                },
                "decoherence_rate": np.random.exponential(0.01)
            }
            
            result = self.quantum_states.insert_one(quantum_record)
            
            self.logger.info(f"Quantum state stored: {state_hash}")
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Error storing quantum state: {str(e)}")
            raise
    
    def update_performance_metrics(
        self, 
        metrics: ModelPerformanceMetrics
    ) -> str:
        """
        Update comprehensive performance metrics for model monitoring
        
        Args:
            metrics: Performance metrics data
            
        Returns:
            MongoDB document ID
        """
        try:
            metrics_dict = asdict(metrics)
            
            # Upsert based on model version and timestamp (hourly aggregation)
            hour_timestamp = metrics.timestamp.replace(minute=0, second=0, microsecond=0)
            
            result = self.performance_metrics.update_one(
                {
                    "model_version": metrics.model_version,
                    "hour_timestamp": hour_timestamp
                },
                {
                    "$set": metrics_dict,
                    "$inc": {"update_count": 1}
                },
                upsert=True
            )
            
            self.logger.info(f"Performance metrics updated for model: {metrics.model_version}")
            return str(result.upserted_id or result.matched_count)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
            raise
    
    def get_conversation_analytics(
        self, 
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analytics from conversation data
        
        Args:
            time_range_hours: Time range for analytics in hours
            
        Returns:
            Analytics summary dictionary
        """
        try:
            # Calculate time threshold
            time_threshold = datetime.now(timezone.utc) - \
                           datetime.timedelta(hours=time_range_hours)
            
            # Aggregation pipeline for comprehensive analytics
            pipeline = [
                {"$match": {"timestamp": {"$gte": time_threshold}}},
                {"$group": {
                    "_id": None,
                    "total_conversations": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence_score"},
                    "avg_processing_time": {"$avg": "$processing_time_ms"},
                    "quantum_enhancement_usage": {
                        "$avg": {"$cond": [{"$eq": ["$quantum_enhancement_used", True]}, 1, 0]}
                    },
                    "query_types": {"$push": "$query_type"}
                }}
            ]
            
            result = list(self.conversations.aggregate(pipeline))
            
            if not result:
                return {"error": "No data available for the specified time range"}
            
            analytics = result[0]
            
            # Process query types distribution
            query_type_counts = {}
            for qt in analytics["query_types"]:
                query_type_counts[qt] = query_type_counts.get(qt, 0) + 1
            
            # Calculate additional metrics
            performance_metrics = {
                "total_conversations": analytics["total_conversations"],
                "average_confidence_score": round(analytics["avg_confidence"], 3),
                "average_processing_time_ms": round(analytics["avg_processing_time"], 2),
                "quantum_enhancement_usage_rate": round(analytics["quantum_enhancement_usage"], 3),
                "query_type_distribution": query_type_counts,
                "time_range_hours": time_range_hours,
                "generated_at": datetime.now(timezone.utc)
            }
            
            # Cache analytics for future use
            self.analytics_cache.replace_one(
                {"cache_key": f"analytics_{time_range_hours}h"},
                {
                    "cache_key": f"analytics_{time_range_hours}h",
                    "data": performance_metrics,
                    "timestamp": datetime.now(timezone.utc)
                },
                upsert=True
            )
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error generating analytics: {str(e)}")
            return {"error": str(e)}
    
    def store_large_model_data(
        self, 
        data: bytes, 
        filename: str,
        metadata: Dict[str, Any]
    ) -> ObjectId:
        """
        Store large model data using GridFS for efficient storage
        
        Args:
            data: Binary data to store
            filename: Name of the file
            metadata: Additional metadata
            
        Returns:
            GridFS file ID
        """
        try:
            file_id = self.fs.put(
                data,
                filename=filename,
                metadata={
                    **metadata,
                    "upload_timestamp": datetime.now(timezone.utc),
                    "size_bytes": len(data)
                }
            )
            
            self.logger.info(f"Large data stored: {filename} ({len(data)} bytes)")
            return file_id
            
        except Exception as e:
            self.logger.error(f"Error storing large data: {str(e)}")
            raise
    
    def get_model_evolution_timeline(self) -> List[Dict[str, Any]]:
        """
        Generate model evolution timeline for research and development insights
        
        Returns:
            Timeline of model versions and performance improvements
        """
        try:
            pipeline = [
                {"$group": {
                    "_id": "$model_version",
                    "first_appearance": {"$min": "$timestamp"},
                    "last_activity": {"$max": "$timestamp"},
                    "total_interactions": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence_score"},
                    "quantum_usage": {
                        "$avg": {"$cond": [{"$eq": ["$quantum_enhancement_used", True]}, 1, 0]}
                    }
                }},
                {"$sort": {"first_appearance": 1}}
            ]
            
            timeline = list(self.conversations.aggregate(pipeline))
            
            # Enhance with performance deltas
            enhanced_timeline = []
            prev_confidence = None
            
            for entry in timeline:
                performance_delta = None
                if prev_confidence is not None:
                    performance_delta = entry["avg_confidence"] - prev_confidence
                
                enhanced_entry = {
                    **entry,
                    "performance_improvement": performance_delta,
                    "days_active": (entry["last_activity"] - entry["first_appearance"]).days,
                }
                enhanced_timeline.append(enhanced_entry)
                prev_confidence = entry["avg_confidence"]
            
            return enhanced_timeline
            
        except Exception as e:
            self.logger.error(f"Error generating model timeline: {str(e)}")
            return []
    
    async def async_batch_store_conversations(
        self, 
        conversations: List[ConversationRecord]
    ) -> List[str]:
        """
        Asynchronously store multiple conversations for high-throughput scenarios
        
        Args:
            conversations: List of conversation records
            
        Returns:
            List of document IDs
        """
        if not self.enable_async:
            raise RuntimeError("Async operations not enabled")
        
        try:
            # Prepare batch documents
            conversation_docs = []
            embedding_docs = []
            
            for conv in conversations:
                record_dict = asdict(conv)
                query_hash = hashlib.sha256(conv.user_query.encode()).hexdigest()
                record_dict['query_hash'] = query_hash
                conversation_docs.append(record_dict)
                
                embedding_docs.append({
                    "conversation_id": conv.conversation_id,
                    "query_hash": query_hash,
                    "embedding": conv.vector_embedding,
                    "query_type": conv.query_type.value,
                    "timestamp": conv.timestamp
                })
            
            # Batch insert operations
            conv_result = await self.async_db.conversations.insert_many(conversation_docs)
            embed_result = await self.async_db.vector_embeddings.insert_many(embedding_docs)
            
            self.logger.info(f"Batch stored {len(conversations)} conversations asynchronously")
            return [str(oid) for oid in conv_result.inserted_ids]
            
        except Exception as e:
            self.logger.error(f"Error in async batch store: {str(e)}")
            raise
    
    def close_connections(self):
        """Close all database connections"""
        self.client.close()
        if self.enable_async:
            self.async_client.close()
        self.logger.info("MongoDB connections closed")

# Utility functions for integration
def create_conversation_record(
    user_query: str,
    model_response: str,
    query_type: QueryType,
    confidence_score: float,
    processing_time_ms: int,
    quantum_enhancement_used: bool,
    vector_embedding: List[float],
    model_version: str = "quantum-illuminator-4b",
    additional_metadata: Optional[Dict[str, Any]] = None
) -> ConversationRecord:
    """
    Factory function to create structured conversation records
    
    Args:
        user_query: Original user question/prompt
        model_response: Generated model response
        query_type: Classification of query type
        confidence_score: Model confidence (0-1)
        processing_time_ms: Time taken to generate response
        quantum_enhancement_used: Whether quantum features were used
        vector_embedding: Semantic embedding of the query
        model_version: Version identifier of the model
        additional_metadata: Any additional data to store
        
    Returns:
        ConversationRecord instance ready for MongoDB storage
    """
    
    conversation_id = hashlib.sha256(
        f"{user_query}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]
    
    performance_metrics = {
        "tokens_generated": len(model_response.split()),
        "query_complexity": len(user_query.split()),
        "response_coherence": confidence_score,
        "quantum_circuit_depth": 6 if quantum_enhancement_used else 0
    }
    
    return ConversationRecord(
        conversation_id=conversation_id,
        user_query=user_query,
        model_response=model_response,
        query_type=query_type,
        confidence_score=confidence_score,
        processing_time_ms=processing_time_ms,
        quantum_enhancement_used=quantum_enhancement_used,
        vector_embedding=vector_embedding,
        metadata=additional_metadata or {},
        timestamp=datetime.now(timezone.utc),
        model_version=model_version,
        performance_metrics=performance_metrics
    )
