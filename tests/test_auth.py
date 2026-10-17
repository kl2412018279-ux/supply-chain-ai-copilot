import pytest

from src.auth.users import (
    InMemoryUserStore,
    LoginError,
    LoginThrottle,
    Role,
    authenticate,
    check_password,
    demo_user,
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


def test_demo_users_are_read_only():
    sales = demo_user("sales")
    assert sales.role is Role.sales and sales.demo
    assert not sales.can_change_shared_data
    assert sales.can_open("quote") and not sales.can_open("data_import")
    assert demo_user().role is Role.manager and demo_user().can_open("evals")


def test_no_accounts_without_admin_password(store):
    assert store.usernames() == []  # visitors don't log in; only an admin account exists


def test_admin_sign_in(monkeypatch):
    import dataclasses

    from src.auth import users
    from src.config import get_settings

    cfg = dataclasses.replace(get_settings(), admin_password="Adm1n-pass")
    monkeypatch.setattr(users, "get_settings", lambda: cfg)
    s = InMemoryUserStore()
    seed_accounts(s)
    assert s.usernames() == ["admin"]
    admin = authenticate(" Admin", "Adm1n-pass", s)
    assert admin.role is Role.admin and not admin.demo
    assert admin.can_change_shared_data and admin.can_open("evals")
    with pytest.raises(LoginError, match="Wrong username or password"):
        authenticate("admin", "nope", s)
    with pytest.raises(LoginError, match="Wrong username or password"):
        authenticate("ghost", "Adm1n-pass", s)
    with pytest.raises(LoginError, match="Enter a username"):
        authenticate("", "", s)

    # Changing the secret changes the password.
    cfg2 = dataclasses.replace(cfg, admin_password="new-pass-2")
    monkeypatch.setattr(users, "get_settings", lambda: cfg2)
    seed_accounts(s)
    assert authenticate("admin", "new-pass-2", s).role is Role.admin


def test_throttle_locks_after_repeated_failures():
    t = LoginThrottle(max_attempts=3, lockout_s=60)
    for i in range(3):
        assert t.locked_for("x", now=100 + i) == 0
        t.failed("x", now=100 + i)
    assert t.locked_for("x", now=103) > 0
    assert t.locked_for("x", now=200) == 0  # lockout expires
    t.reset("x")
    assert t.locked_for("x", now=103) == 0
