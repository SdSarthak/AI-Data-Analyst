"""
User authentication and session management
"""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import bcrypt
from config.settings import APP_SECRET_KEY, SESSION_TIMEOUT_MINUTES
from utils.logger import setup_logger
from utils.errors import AuthenticationError

logger = setup_logger(__name__)


class UserManager:
    """
    User authentication and management
    """
    
    # In production, use a proper database
    _users_db = {}
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using bcrypt
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Plain text password
            password_hash: Hashed password
            
        Returns:
            True if password matches
        """
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    @classmethod
    def create_user(cls, username: str, email: str, password: str) -> bool:
        """
        Create a new user
        
        Args:
            username: Username
            email: Email address
            password: Password
            
        Returns:
            True if user created successfully
            
        Raises:
            AuthenticationError: If user already exists
        """
        if username in cls._users_db:
            raise AuthenticationError(f"User '{username}' already exists")
        
        cls._users_db[username] = {
            'email': email,
            'password_hash': cls.hash_password(password),
            'created_at': datetime.now(),
            'last_login': None
        }
        
        logger.info(f"User '{username}' created successfully")
        return True
    
    @classmethod
    def authenticate(cls, username: str, password: str) -> bool:
        """
        Authenticate user credentials
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if credentials are valid
            
        Raises:
            AuthenticationError: If authentication fails
        """
        if username not in cls._users_db:
            raise AuthenticationError("Invalid username or password")
        
        user = cls._users_db[username]
        if not cls.verify_password(password, user['password_hash']):
            raise AuthenticationError("Invalid username or password")
        
        # Update last login
        user['last_login'] = datetime.now()
        logger.info(f"User '{username}' authenticated successfully")
        return True
    
    @classmethod
    def get_user(cls, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user information
        
        Args:
            username: Username
            
        Returns:
            User information dictionary or None
        """
        if username in cls._users_db:
            user = cls._users_db[username].copy()
            user.pop('password_hash', None)  # Don't return password hash
            return user
        return None


class SessionManager:
    """
    Session management for authenticated users
    """
    
    _sessions = {}
    
    @staticmethod
    def create_session(username: str) -> str:
        """
        Create a new session for user
        
        Args:
            username: Username
            
        Returns:
            Session token
        """
        session_token = secrets.token_urlsafe(32)
        SessionManager._sessions[session_token] = {
            'username': username,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'query_history': [],
            'favorites': []
        }
        logger.info(f"Session created for user '{username}'")
        return session_token
    
    @staticmethod
    def validate_session(session_token: str) -> bool:
        """
        Validate if session is still active
        
        Args:
            session_token: Session token to validate
            
        Returns:
            True if session is valid
        """
        if session_token not in SessionManager._sessions:
            return False
        
        session = SessionManager._sessions[session_token]
        elapsed = datetime.now() - session['last_activity']
        
        if elapsed > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            del SessionManager._sessions[session_token]
            logger.info("Session expired due to inactivity")
            return False
        
        # Update last activity
        session['last_activity'] = datetime.now()
        return True
    
    @staticmethod
    def get_session_data(session_token: str) -> Optional[Dict[str, Any]]:
        """
        Get session data
        
        Args:
            session_token: Session token
            
        Returns:
            Session data or None
        """
        if SessionManager.validate_session(session_token):
            return SessionManager._sessions[session_token]
        return None
    
    @staticmethod
    def add_to_history(session_token: str, query: str):
        """
        Add query to user's history
        
        Args:
            session_token: Session token
            query: Query string
        """
        if session_token in SessionManager._sessions:
            SessionManager._sessions[session_token]['query_history'].append({
                'query': query,
                'timestamp': datetime.now()
            })
    
    @staticmethod
    def add_to_favorites(session_token: str, query: str) -> bool:
        """
        Add query to user's favorites
        
        Args:
            session_token: Session token
            query: Query string
            
        Returns:
            True if added successfully
        """
        if session_token in SessionManager._sessions:
            SessionManager._sessions[session_token]['favorites'].append({
                'query': query,
                'added_at': datetime.now()
            })
            return True
        return False
    
    @staticmethod
    def destroy_session(session_token: str):
        """
        Destroy a session
        
        Args:
            session_token: Session token
        """
        if session_token in SessionManager._sessions:
            del SessionManager._sessions[session_token]
            logger.info("Session destroyed")
