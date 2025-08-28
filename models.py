# from sqlalchemy import Column, Integer, String, Text, Boolean, TIMESTAMP, ForeignKey, func
# from sqlalchemy.orm import relationship
# from datetime import datetime
# from database import Base
# import pytz

# IST = pytz.timezone('Asia/Kolkata')
# def ist_now():
#     """Default function for IST timestamps"""
#     return datetime.now(IST)

# CHARACTER_CATEGORIES = {
#     'entertainment_arts': 'Entertainment & Arts',
#     'leaders_historical': 'Leaders & Historical', 
#     'sports_champions': 'Sports Champions',
#     'innovators_visionaries': 'Innovators & Visionaries',
#     'spiritual_social': 'Spiritual & Social',
#     'fictional_anime': 'Fictional & Anime'
# }

# class User(Base):
#     __tablename__ = "users"
    
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(100), unique=True, nullable=False, index=True)
#     email = Column(String(255), unique=True, nullable=True, index=True)  # NEW: For OAuth
#     name = Column(String(255), nullable=True)  # NEW: Full name from OAuth
#     password_hash = Column(String(255), nullable=True)  # MODIFIED: Nullable for OAuth users
    
#     # OAuth fields
#     oauth_provider = Column(String(50), nullable=True)  # NEW: 'google', 'facebook', etc.
#     oauth_id = Column(String(255), nullable=True)  # NEW: OAuth provider's user ID
#     is_oauth_user = Column(Boolean, default=False, nullable=False)  # NEW: Flag for OAuth users
    
#     # Timestamps
#     # Add these new fields
#     registration_ip = Column(String(45), nullable=True)  # IPv6 can be up to 45 chars
#     last_login_ip = Column(String(45), nullable=True)
#     created_at = Column(DateTime, default=ist_now, nullable=False)
#     last_login = Column(DateTime, nullable=True)  # NEW: Track last login
    
#     # Relationships
#     sessions = relationship("ConversationSession", back_populates="user")
    
#     def __repr__(self):
#         return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
    
    
# class Character(Base):
#     __tablename__ = "characters"
    
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(100), unique=True, nullable=False, index=True)
#     prompt = Column(Text, nullable=False)
#     img = Column(String(255))
#     description = Column(Text)
#     native_language = Column(String(50), nullable=True, default='english')
#     is_multilingual = Column(Boolean, default=True, nullable=False)
#     category = Column(String(50), nullable=True, default='entertainment_arts', index=True)
#     created_at = Column(TIMESTAMP, server_default=func.now())
    
#     # Relationships
#     knowledge_entries = relationship("CharacterKnowledge", back_populates="character")
#     sessions = relationship("ConversationSession", back_populates="character_obj")

# class CharacterKnowledge(Base):
#     __tablename__ = "character_knowledge"
    
#     id = Column(Integer, primary_key=True, index=True)
#     character_id = Column(Integer, ForeignKey("characters.id"), nullable=False, index=True)
#     content = Column(Text, nullable=False)
#     language = Column(String(50), default='english', nullable=False, index=True)
#     embedding = Column(Text)  # Optional: for future vector search
#     created_at = Column(TIMESTAMP, server_default=func.now())
    
#     # Relationships
#     character = relationship("Character", back_populates="knowledge_entries")

# class ConversationSession(Base):
#     __tablename__ = "conversation_sessions"
    
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
#     character_name = Column(String(50), nullable=False, index=True)
#     character_id = Column(Integer, ForeignKey("characters.id"), index=True)
#     primary_language = Column(String(50), default='english', nullable=False, index=True)
#     created_at = Column(TIMESTAMP, server_default=func.now())
    
#     # Relationships
#     user = relationship("User", back_populates="sessions")
#     character_obj = relationship("Character", back_populates="sessions")
#     messages = relationship("ConversationMessage", back_populates="session")

# class ConversationMessage(Base):
#     __tablename__ = "conversation_messages"
    
#     id = Column(Integer, primary_key=True, index=True)
#     session_id = Column(Integer, ForeignKey("conversation_sessions.id"), nullable=False, index=True)
#     role = Column(String(50), nullable=False)  # 'user' or character name
#     content = Column(Text, nullable=False)
#     language = Column(String(50), default='english', nullable=False, index=True)
#     audio_base64 = Column(Text, nullable=True)  # NEW: Store base64 encoded audio data
#     timestamp = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    
#     # Relationships
#     session = relationship("ConversationSession", back_populates="messages")


from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base
import pytz

IST = pytz.timezone('Asia/Kolkata')

def ist_now():
    """Default function for IST timestamps - returns naive datetime for SQLAlchemy"""
    return datetime.now(IST).replace(tzinfo=None)  # Remove timezone info for SQLAlchemy

CHARACTER_CATEGORIES = {
    'entertainment_arts': 'Entertainment & Arts',
    'leaders_historical': 'Leaders & Historical', 
    'sports_champions': 'Sports Champions',
    'innovators_visionaries': 'Innovators & Visionaries',
    'spiritual_social': 'Spiritual & Social',
    'fictional_anime': 'Fictional & Anime'
}

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    mobile_number = Column(String(20), unique=True, nullable=True, index=True)  # NEW: Mobile number field
    name = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth users
    
    # OAuth fields
    oauth_provider = Column(String(50), nullable=True)
    oauth_id = Column(String(255), nullable=True)
    is_oauth_user = Column(Boolean, default=False, nullable=False)
    
    # IP tracking fields
    registration_ip = Column(String(45), nullable=True)  # IPv6 can be up to 45 chars
    last_login_ip = Column(String(45), nullable=True)
    
    # Timestamps - ALL using DateTime with IST
    created_at = Column(DateTime, default=ist_now, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    sessions = relationship("ConversationSession", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}', mobile='{self.mobile_number}')>"

class Character(Base):
    __tablename__ = "characters"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    img = Column(String(255))
    description = Column(Text)
    native_language = Column(String(50), nullable=True, default='english')
    is_multilingual = Column(Boolean, default=True, nullable=False)
    category = Column(String(50), nullable=True, default='entertainment_arts', index=True)
    
    # Fixed: Use DateTime with IST instead of TIMESTAMP
    created_at = Column(DateTime, default=ist_now, nullable=False)
    
    # Relationships
    knowledge_entries = relationship("CharacterKnowledge", back_populates="character")
    sessions = relationship("ConversationSession", back_populates="character_obj")

class CharacterKnowledge(Base):
    __tablename__ = "character_knowledge"
    
    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    language = Column(String(50), default='english', nullable=False, index=True)
    embedding = Column(Text)  # Optional: for future vector search
    
    # Fixed: Use DateTime with IST instead of TIMESTAMP
    created_at = Column(DateTime, default=ist_now, nullable=False)
    
    # Relationships
    character = relationship("Character", back_populates="knowledge_entries")

class ConversationSession(Base):
    __tablename__ = "conversation_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    character_name = Column(String(50), nullable=False, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"), index=True)
    primary_language = Column(String(50), default='english', nullable=False, index=True)
    
    # Fixed: Use DateTime with IST instead of TIMESTAMP
    created_at = Column(DateTime, default=ist_now, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    character_obj = relationship("Character", back_populates="sessions")
    messages = relationship("ConversationMessage", back_populates="session")

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("conversation_sessions.id"), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # 'user' or character name
    content = Column(Text, nullable=False)
    language = Column(String(50), default='english', nullable=False, index=True)
    audio_base64 = Column(Text, nullable=True)  # Store base64 encoded audio data
    
    # Fixed: Use DateTime with IST instead of TIMESTAMP
    timestamp = Column(DateTime, default=ist_now, nullable=False)
    
    # Relationships
    session = relationship("ConversationSession", back_populates="messages")