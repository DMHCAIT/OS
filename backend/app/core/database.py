"""
Database connection and configuration
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global database client
mongo_client: Optional[AsyncIOMotorClient] = None
database = None


async def connect_to_mongo():
    """Create database connection"""
    global mongo_client, database
    
    try:
        # Create MongoDB client
        mongo_client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            maxPoolSize=10,
            minPoolSize=5,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        await mongo_client.admin.command('ping')
        
        # Get database
        database = mongo_client[settings.DATABASE_NAME]
        
        logger.info(f"Connected to MongoDB: {settings.DATABASE_NAME}")
        
        # Create indexes
        await create_indexes()
        
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close database connection"""
    global mongo_client
    
    if mongo_client:
        mongo_client.close()
        logger.info("Disconnected from MongoDB")


async def get_database():
    """Get database instance"""
    global database
    return database


async def create_indexes():
    """Create database indexes for optimal performance"""
    global database
    
    if not database:
        return
    
    try:
        # Leads collection indexes
        await database.leads.create_index("email", unique=True)
        await database.leads.create_index("phone")
        await database.leads.create_index("status")
        await database.leads.create_index("score")
        await database.leads.create_index("created_at")
        await database.leads.create_index("company")
        
        # Calls collection indexes
        await database.calls.create_index("lead_id")
        await database.calls.create_index("status")
        await database.calls.create_index("created_at")
        await database.calls.create_index("duration")
        
        # Conversations collection indexes
        await database.conversations.create_index("call_id")
        await database.conversations.create_index("lead_id")
        await database.conversations.create_index("created_at")
        
        # Users collection indexes
        await database.users.create_index("email", unique=True)
        await database.users.create_index("username", unique=True)
        await database.users.create_index("role")
        
        # Analytics collection indexes
        await database.analytics.create_index("date")
        await database.analytics.create_index("metric_type")
        await database.analytics.create_index("user_id")
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")


# Database collections enum
class Collections:
    LEADS = "leads"
    CALLS = "calls"
    CONVERSATIONS = "conversations"
    USERS = "users"
    ANALYTICS = "analytics"
    CAMPAIGNS = "campaigns"
    TEMPLATES = "templates"
    FILES = "files"