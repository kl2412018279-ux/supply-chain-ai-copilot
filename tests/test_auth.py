import pytest

from src.auth.users import (
    DEMO_PASSWORD,
    InMemoryUserStore,
    LoginError,
    LoginThrottle,
    Role,
    User,
    authenticate,
    check_password,
    hash_password,
    seed_accounts,
)


@pytest.fixture
def store():
    s = InMemoryUserStore()
    seed_accounts(s)
    return s


def test_hash_round_trip_and_no_plaintext():
    h = hash_password("secret-1")
    assert h != "secret-1" and h.startswith("$2b$")
    assert check_password("secret-1", h)
    assert not check_password("secret-2", h)
    assert not check_password("x", "not-a-hash")


def test_demo_accounts_seeded(store):
    assert {"manager", "ops", "sales"} <= set(store.usernames())
    user = authenticate("Sales ", DEMO_PASSWORD, store)  # case and spaces tolerated
    assert user.role is Role.sales and user.demo
    assert not user.can_change_shared_data
    assert user.can_open("quote") and not user.can_open("data_import")


def test_wrong_password_and_unknown_user(store):
    with pytest.raises(LoginError, match="Wrong username or password"):
        authenticate("manager", "nope", store)
    with pytest.raises(LoginError, match="Wrong username or password"):
        authenticate("ghost", DEMO_PASSWORD, store)
    with pytest.raises(LoginError, match="Enter a username"):
        authenticate("", "", store)


def test_admin_from_settings(monkeypatch):
    import dataclasses

    from src.auth import users
    from src.config import get_settings

    cfg = dataclasses.replace(get_settings(), admin_password="Adm1n-pass", demo_accounts=False)
    monkeypatch.setattr(users, "get_settings", lambda: cfg)
    s = InMemoryUserStore()
    s.upsert(User("ops", Role.ops, "old demo", True), hash_password(DEMO_PASSWORD))
    seed_accounts(s)
    assert s.usernames() == ["admin"]  # demo accounts removed when switched off
    admin = authenticate("admin", "Adm1n-pass", s)
    assert admin.role is Role.admin and admin.can_change_shared_data and admin.can_open("evals")


def test_throttle_locks_after_repeated_failures():
    t = LoginThrottle(max_attempts=3, lockout_s=60)
    for i in range(3):
        assert t.locked_for("x", now=100 + i) == 0
        t.failed("x", now=100 + i)
    assert t.locked_for("x", now=103) > 0
    assert t.locked_for("x", now=200) == 0  # lockout expires
    t.reset("x")
    assert t.locked_for("x", now=103) == 0
