"""Tests for text2sql authentication and session handling."""
import json
from datetime import datetime, timedelta

import pytest

from src.auth import MIN_PASSWORD_LENGTH, SessionManager, UserManager
from utils.errors import AuthenticationError


@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    """Point the user store at a temporary file for every test."""
    UserManager.set_store_path(tmp_path / "users.json")
    SessionManager._sessions.clear()
    yield tmp_path / "users.json"
    UserManager.set_store_path(tmp_path / "users.json")


class TestPasswordHashing:
    def test_hash_is_not_the_password(self):
        assert UserManager.hash_password("hunter2000") != "hunter2000"

    def test_hashes_are_salted(self):
        assert UserManager.hash_password("hunter2000") != UserManager.hash_password("hunter2000")

    def test_verify_accepts_the_right_password(self):
        assert UserManager.verify_password("hunter2000", UserManager.hash_password("hunter2000"))

    def test_verify_rejects_the_wrong_password(self):
        assert not UserManager.verify_password("nope", UserManager.hash_password("hunter2000"))

    def test_corrupt_hash_returns_false_instead_of_raising(self):
        assert UserManager.verify_password("hunter2000", "not-a-bcrypt-hash") is False

    def test_empty_inputs_return_false(self):
        assert UserManager.verify_password("", "") is False


class TestCreateUser:
    def test_creates_an_account(self):
        assert UserManager.create_user("alice", "alice@example.com", "hunter2000") is True
        assert UserManager.user_count() == 1

    def test_rejects_duplicate_username(self):
        UserManager.create_user("alice", "alice@example.com", "hunter2000")
        with pytest.raises(AuthenticationError, match="already exists"):
            UserManager.create_user("alice", "other@example.com", "hunter2000")

    def test_rejects_blank_username(self):
        with pytest.raises(AuthenticationError):
            UserManager.create_user("   ", "alice@example.com", "hunter2000")

    def test_rejects_invalid_email(self):
        with pytest.raises(AuthenticationError, match="email"):
            UserManager.create_user("alice", "not-an-email", "hunter2000")

    def test_rejects_short_password(self):
        with pytest.raises(AuthenticationError, match=str(MIN_PASSWORD_LENGTH)):
            UserManager.create_user("alice", "alice@example.com", "short")


class TestAuthenticate:
    def test_accepts_correct_credentials(self):
        UserManager.create_user("alice", "alice@example.com", "hunter2000")
        assert UserManager.authenticate("alice", "hunter2000") is True

    def test_rejects_wrong_password(self):
        UserManager.create_user("alice", "alice@example.com", "hunter2000")
        with pytest.raises(AuthenticationError):
            UserManager.authenticate("alice", "wrong-password")

    def test_unknown_user_and_wrong_password_look_the_same(self):
        """The error must not reveal whether an account exists."""
        UserManager.create_user("alice", "alice@example.com", "hunter2000")
        with pytest.raises(AuthenticationError) as unknown:
            UserManager.authenticate("nobody", "hunter2000")
        with pytest.raises(AuthenticationError) as wrong:
            UserManager.authenticate("alice", "wrong-password")
        assert str(unknown.value) == str(wrong.value)

    def test_records_last_login(self):
        UserManager.create_user("alice", "alice@example.com", "hunter2000")
        assert UserManager.get_user("alice")["last_login"] is None
        UserManager.authenticate("alice", "hunter2000")
        assert UserManager.get_user("alice")["last_login"] is not None


class TestPersistence:
    def test_accounts_survive_a_reload(self, isolated_store):
        """Regression: accounts lived only in memory and vanished on restart."""
        UserManager.create_user("alice", "alice@example.com", "hunter2000")
        UserManager.set_store_path(isolated_store)  # simulates a fresh process
        assert UserManager.authenticate("alice", "hunter2000") is True

    def test_store_never_contains_the_plain_password(self, isolated_store):
        UserManager.create_user("alice", "alice@example.com", "hunter2000")
        assert "hunter2000" not in isolated_store.read_text(encoding="utf-8")

    def test_store_contains_a_hash(self, isolated_store):
        UserManager.create_user("alice", "alice@example.com", "hunter2000")
        data = json.loads(isolated_store.read_text(encoding="utf-8"))
        assert data["alice"]["password_hash"].startswith("$2")

    def test_corrupt_store_is_ignored(self, isolated_store):
        isolated_store.write_text("{ not json", encoding="utf-8")
        UserManager.set_store_path(isolated_store)
        assert UserManager.user_count() == 0

    def test_missing_store_is_not_an_error(self, tmp_path):
        UserManager.set_store_path(tmp_path / "nested" / "does-not-exist.json")
        assert UserManager.user_count() == 0


class TestGetUser:
    def test_never_returns_the_password_hash(self):
        UserManager.create_user("alice", "alice@example.com", "hunter2000")
        assert "password_hash" not in UserManager.get_user("alice")

    def test_unknown_user_returns_none(self):
        assert UserManager.get_user("nobody") is None


class TestSessions:
    def test_tokens_are_unique(self):
        assert SessionManager.create_session("alice") != SessionManager.create_session("alice")

    def test_new_session_is_valid(self):
        assert SessionManager.validate_session(SessionManager.create_session("alice")) is True

    def test_unknown_token_is_invalid(self):
        assert SessionManager.validate_session("made-up-token") is False

    def test_expired_session_is_dropped(self):
        token = SessionManager.create_session("alice")
        SessionManager._sessions[token]["last_activity"] = datetime.now() - timedelta(days=1)
        assert SessionManager.validate_session(token) is False
        assert token not in SessionManager._sessions

    def test_history_is_recorded(self):
        token = SessionManager.create_session("alice")
        SessionManager.add_to_history(token, "how many orders?")
        assert SessionManager.get_session_data(token)["query_history"][0]["query"] == (
            "how many orders?"
        )

    def test_history_on_unknown_session_is_a_no_op(self):
        assert SessionManager.add_to_history("made-up", "q") is False

    def test_favourites_are_deduplicated(self):
        token = SessionManager.create_session("alice")
        assert SessionManager.add_to_favorites(token, "SELECT 1") is True
        assert SessionManager.add_to_favorites(token, "SELECT 1") is False
        assert len(SessionManager.get_session_data(token)["favorites"]) == 1

    def test_destroy_session(self):
        token = SessionManager.create_session("alice")
        SessionManager.destroy_session(token)
        assert SessionManager.validate_session(token) is False

    def test_destroy_unknown_session_is_safe(self):
        SessionManager.destroy_session("made-up-token")
