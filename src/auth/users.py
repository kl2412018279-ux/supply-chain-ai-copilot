"""User accounts, password hashing and roles for the dashboard login.

Passwords are stored only as bcrypt hashes. Two kinds of account exist:

* **Demo accounts** (on by default, ``DEMO_ACCOUNTS=false`` to switch off): one per
  role, with a password shown on the login page so recruiters and interviewers can
  look around. Demo users cannot change shared data: their uploads stay in their own
  browser session and they cannot send Telegram alerts.
* **Admin account** (only when ``ADMIN_PASSWORD`` is set, e.g. in Streamlit secrets):
  full access, including saving uploads for everyone and sending alerts. Its
  password is never shown and the hash is refreshed from the secret on every start,
  so changing the secret changes the password.

Accounts live in PostgreSQL (``app_users``) when ``DATABASE_URL`` is set, else in memory.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Protocol

import bcrypt

from src.config import get_settings
from src.db import PgConnection

logger = logging.getLogger(__name__)


class Role(str, Enum):
    admin = "admin"
    manager = "manager"
    ops = "ops"
    sales = "sales"


ROLE_LABELS = {
    Role.admin: "Administrator",
    Role.manager: "Operations manager",
    Role.ops: "Operations executive",
    Role.sales: "Sales executive",
}

# Pages (by file stem) each role can open. Admin and manager see everything.
ALL_PAGES = {
    "control_tower", "copilot", "quote", "planner", "delays", "forecast", "intake",
    "data_import", "alerts", "evals", "hauliers", "documents",
}  # fmt: skip
ROLE_PAGES: dict[Role, set[str]] = {
    Role.admin: ALL_PAGES,
    Role.manager: ALL_PAGES,
    Role.ops: {
        "control_tower",
        "copilot",
        "delays",
        "forecast",
        "intake",
        "data_import",
        "alerts",
        "hauliers",
        "documents",
    },  # fmt: skip
    Role.sales: {
        "control_tower",
        "copilot",
        "quote",
        "planner",
        "forecast",
        "hauliers",
        "documents",
    },  # fmt: skip
}

DEMO_PASSWORD = "muara-demo"
DEMO_ACCOUNTS = [
    ("manager", Role.manager, "Aisyah (Ops Manager)"),
    ("ops", Role.ops, "Ravi (Ops Executive)"),
    ("sales", Role.sales, "Mei Ling (Sales Executive)"),
]

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


@dataclass(frozen=True)
class User:
    username: str
    role: Role
    display_name: str
    demo: bool

    @property
    def role_label(self) -> str:
        return ROLE_LABELS[self.role]

    def can_open(self, page: str) -> bool:
        return page in ROLE_PAGES[self.role]

    @property
    def can_change_shared_data(self) -> bool:
        """Save uploads for everyone, send alerts. Never for public demo accounts."""
        return not self.demo and self.role in {Role.admin, Role.manager, Role.ops}


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def check_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:  # malformed hash
        return False


class UserStore(Protocol):
    def upsert(self, user: User, password_hash: str) -> None: ...
    def get(self, username: str) -> tuple[User, str] | None: ...
    def delete(self, username: str) -> None: ...
    def usernames(self) -> list[str]: ...


class InMemoryUserStore:
    def __init__(self) -> None:
        self._users: dict[str, tuple[User, str]] = {}
        self._lock = threading.Lock()

    def upsert(self, user: User, password_hash: str) -> None:
        with self._lock:
            self._users[user.username] = (user, password_hash)

    def get(self, username: str) -> tuple[User, str] | None:
        return self._users.get(username)

    def delete(self, username: str) -> None:
        with self._lock:
            self._users.pop(username, None)

    def usernames(self) -> list[str]:
        return sorted(self._users)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL,
    display_name  TEXT NOT NULL,
    demo          BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ
)
"""


class PostgresUserStore:
    def __init__(self, dsn: str):
        self._conn = PgConnection(dsn)
        self._conn.execute(SCHEMA_SQL)

    def upsert(self, user: User, password_hash: str) -> None:
        self._conn.execute(
            """INSERT INTO app_users (username, password_hash, role, display_name, demo)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash,
                 role = EXCLUDED.role, display_name = EXCLUDED.display_name,
                 demo = EXCLUDED.demo""",
            (user.username, password_hash, user.role.value, user.display_name, user.demo),
        )

    def get(self, username: str) -> tuple[User, str] | None:
        row = self._conn.execute(
            "SELECT username, role, display_name, demo, password_hash FROM app_users "
            "WHERE username = %s",
            (username,),
        ).fetchone()
        if not row:
            return None
        return User(row[0], Role(row[1]), row[2], bool(row[3])), row[4]

    def delete(self, username: str) -> None:
        self._conn.execute("DELETE FROM app_users WHERE username = %s", (username,))

    def usernames(self) -> list[str]:
        rows = self._conn.execute("SELECT username FROM app_users ORDER BY 1").fetchall()
        return [r[0] for r in rows]

    def touch(self, username: str) -> None:
        self._conn.execute(
            "UPDATE app_users SET last_login_at = now() WHERE username = %s", (username,)
        )


def seed_accounts(store: UserStore) -> None:
    """Create / refresh the demo accounts and the admin account from settings."""
    settings = get_settings()
    for username, role, name in DEMO_ACCOUNTS:
        if settings.demo_accounts:
            existing = store.get(username)
            if existing is None or not check_password(DEMO_PASSWORD, existing[1]):
                store.upsert(User(username, role, name, demo=True), hash_password(DEMO_PASSWORD))
        else:
            store.delete(username)
    if settings.admin_password:
        admin = settings.admin_username
        existing = store.get(admin)
        if existing is None or not check_password(settings.admin_password, existing[1]):
            store.upsert(
                User(admin, Role.admin, "Administrator", demo=False),
                hash_password(settings.admin_password),
            )


@lru_cache
def get_user_store() -> UserStore:
    dsn = get_settings().database_url
    store: UserStore
    if dsn:
        try:
            store = PostgresUserStore(dsn)
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s); keeping accounts in memory.", exc)
            store = InMemoryUserStore()
    else:
        store = InMemoryUserStore()
    seed_accounts(store)
    return store


class LoginThrottle:
    """Lock a username out for a minute after repeated wrong passwords."""

    def __init__(self, max_attempts: int = MAX_ATTEMPTS, lockout_s: float = LOCKOUT_SECONDS):
        self.max_attempts, self.lockout_s = max_attempts, lockout_s
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def locked_for(self, username: str, now: float | None = None) -> float:
        now = now or time.monotonic()
        recent = [t for t in self._fails.get(username, []) if now - t < self.lockout_s]
        if len(recent) < self.max_attempts:
            return 0.0
        return self.lockout_s - (now - recent[-self.max_attempts])

    def failed(self, username: str, now: float | None = None) -> None:
        with self._lock:
            self._fails.setdefault(username, []).append(now or time.monotonic())

    def reset(self, username: str) -> None:
        with self._lock:
            self._fails.pop(username, None)


_throttle = LoginThrottle()


@lru_cache
def _dummy_hash() -> str:
    return hash_password("not-a-real-account")


class LoginError(Exception):
    pass


def authenticate(username: str, password: str, store: UserStore | None = None) -> User:
    """Return the user for valid credentials, else raise ``LoginError``."""
    username = username.strip().lower()
    if not username or not password:
        raise LoginError("Enter a username and password.")
    wait = _throttle.locked_for(username)
    if wait > 0:
        raise LoginError(f"Too many attempts. Try again in {wait:.0f} seconds.")
    store = store or get_user_store()
    found = store.get(username)
    # Check a hash even for unknown users so the response time doesn't reveal them.
    hashed = found[1] if found else _dummy_hash()
    if not check_password(password, hashed) or found is None:
        _throttle.failed(username)
        raise LoginError("Wrong username or password.")
    _throttle.reset(username)
    if isinstance(store, PostgresUserStore):
        store.touch(username)
    return found[0]
