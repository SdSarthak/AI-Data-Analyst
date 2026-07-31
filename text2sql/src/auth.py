"""
User authentication and session management.

Accounts are persisted to a JSON file (only the bcrypt hash is stored, never
the password) so signing up once survives a restart of the Streamlit server.
Set ``USER_STORE_PATH`` to move the file; the default lives outside the
repository tree that gets committed.
"""
import json
import secrets
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import bcrypt

from config.settings import SESSION_TIMEOUT_MINUTES, USER_STORE_PATH
from utils.errors import AuthenticationError
from utils.logger import setup_logger

logger = setup_logger(__name__)

MIN_PASSWORD_LENGTH = 8


class UserManager:
    """User registration and credential verification."""

    _users_db: Dict[str, Dict[str, Any]] = {}
    _loaded = False
    _lock = threading.Lock()
    _store_path = Path(USER_STORE_PATH)

    # --- persistence -----------------------------------------------------

    @classmethod
    def set_store_path(cls, path) -> None:
        """Point the store at a different file and reload. Used by tests."""
        with cls._lock:
            cls._store_path = Path(path)
            cls._users_db = {}
            cls._loaded = False

    @classmethod
    def _load(cls) -> None:
        """Read the user store from disk once per process."""
        if cls._loaded:
            return
        cls._loaded = True
        try:
            raw = cls._store_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Could not read user store: %s", exc)
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("User store is not valid JSON, ignoring it: %s", exc)
            return

        if isinstance(data, dict):
            cls._users_db = {
                name: record
                for name, record in data.items()
                if isinstance(record, dict) and record.get("password_hash")
            }
            logger.info("Loaded %d user(s) from the store", len(cls._users_db))

    @classmethod
    def _save(cls) -> None:
        """Write the user store back to disk."""
        try:
            cls._store_path.parent.mkdir(parents=True, exist_ok=True)
            cls._store_path.write_text(
                json.dumps(cls._users_db, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not persist user store: %s", exc)

    # --- passwords -------------------------------------------------------

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password with bcrypt."""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Check a password against a bcrypt hash.

        Returns ``False`` rather than raising when the stored hash is corrupt,
        so a damaged record cannot crash the login screen.
        """
        if not password or not password_hash:
            return False
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except (ValueError, TypeError):
            logger.warning("Stored password hash is unreadable")
            return False

    # --- accounts --------------------------------------------------------

    @classmethod
    def create_user(cls, username: str, email: str, password: str) -> bool:
        """
        Register a new account.

        Args:
            username: Desired username.
            email: Email address.
            password: Plain text password, at least ``MIN_PASSWORD_LENGTH``.

        Returns:
            ``True`` on success.

        Raises:
            AuthenticationError: If the input is invalid or the name is taken.
        """
        username = (username or "").strip()
        email = (email or "").strip()

        if not username:
            raise AuthenticationError("Username is required")
        if not email or "@" not in email:
            raise AuthenticationError("A valid email address is required")
        if not password or len(password) < MIN_PASSWORD_LENGTH:
            raise AuthenticationError(
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
            )

        with cls._lock:
            cls._load()
            if username in cls._users_db:
                raise AuthenticationError(f"User '{username}' already exists")

            cls._users_db[username] = {
                "email": email,
                "password_hash": cls.hash_password(password),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "last_login": None,
            }
            cls._save()

        logger.info("User '%s' created successfully", username)
        return True

    @classmethod
    def authenticate(cls, username: str, password: str) -> bool:
        """
        Verify credentials.

        The same message is returned whether the username or the password was
        wrong, so the response does not reveal which accounts exist.

        Raises:
            AuthenticationError: If the credentials do not match.
        """
        username = (username or "").strip()
        with cls._lock:
            cls._load()
            user = cls._users_db.get(username)
            valid = user is not None and cls.verify_password(password, user["password_hash"])
            if not valid:
                logger.info("Failed login attempt for '%s'", username)
                raise AuthenticationError("Invalid username or password")

            user["last_login"] = datetime.now().isoformat(timespec="seconds")
            cls._save()

        logger.info("User '%s' authenticated successfully", username)
        return True

    @classmethod
    def get_user(cls, username: str) -> Optional[Dict[str, Any]]:
        """Return public user details, never the password hash."""
        with cls._lock:
            cls._load()
            user = cls._users_db.get((username or "").strip())
            if user is None:
                return None
            public = dict(user)
            public.pop("password_hash", None)
            return public

    @classmethod
    def user_count(cls) -> int:
        """Return how many accounts are registered."""
        with cls._lock:
            cls._load()
            return len(cls._users_db)


class SessionManager:
    """In-memory session store for the running Streamlit process."""

    _sessions: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def create_session(username: str) -> str:
        """Create a session and return its token."""
        session_token = secrets.token_urlsafe(32)
        now = datetime.now()
        SessionManager._sessions[session_token] = {
            "username": username,
            "created_at": now,
            "last_activity": now,
            "query_history": [],
            "favorites": [],
        }
        logger.info("Session created for user '%s'", username)
        return session_token

    @staticmethod
    def validate_session(session_token: str) -> bool:
        """
        Check a session is known and has not timed out.

        Expired sessions are removed as a side effect.
        """
        session = SessionManager._sessions.get(session_token)
        if session is None:
            return False

        if datetime.now() - session["last_activity"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            del SessionManager._sessions[session_token]
            logger.info("Session expired due to inactivity")
            return False

        session["last_activity"] = datetime.now()
        return True

    @staticmethod
    def get_session_data(session_token: str) -> Optional[Dict[str, Any]]:
        """Return the session payload, or ``None`` when it is not valid."""
        if SessionManager.validate_session(session_token):
            return SessionManager._sessions[session_token]
        return None

    @staticmethod
    def add_to_history(session_token: str, query: str) -> bool:
        """Record a question in the session's history."""
        session = SessionManager._sessions.get(session_token)
        if session is None:
            return False
        session["query_history"].append({"query": query, "timestamp": datetime.now()})
        return True

    @staticmethod
    def add_to_favorites(session_token: str, query: str) -> bool:
        """
        Save a query to the session's favourites.

        Duplicates are ignored so clicking twice does not add it twice.
        """
        session = SessionManager._sessions.get(session_token)
        if session is None:
            return False
        if any(item["query"] == query for item in session["favorites"]):
            return False
        session["favorites"].append({"query": query, "added_at": datetime.now()})
        return True

    @staticmethod
    def destroy_session(session_token: str) -> None:
        """Forget a session."""
        if SessionManager._sessions.pop(session_token, None) is not None:
            logger.info("Session destroyed")
