import sqlite3
from datetime import datetime
from auth.models import User


def test_user_from_row():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE users(id,email,hashed_pw,role,provider,
                    provider_id,invited_by,created_at)""")
    conn.execute("INSERT INTO users VALUES(?,?,?,?,?,?,?,?)",
                 ("u1","x@y.com",None,"admin","google","gid",None,"2026-01-01"))
    row = conn.execute("SELECT * FROM users").fetchone()
    u = User.from_row(row)
    assert u.id == "u1"
    assert u.role == "admin"
    assert u.provider == "google"
    assert u.hashed_pw is None
