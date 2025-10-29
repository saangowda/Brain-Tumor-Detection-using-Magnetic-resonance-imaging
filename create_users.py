import sqlite3
from werkzeug.security import generate_password_hash

# Connect to (or create) users.db
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create the users table
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')

# Drop old history table if exists (to fix column mismatch)
c.execute('DROP TABLE IF EXISTS history')

# Create the history table with 'username' column
c.execute('''
    CREATE TABLE history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        filename TEXT NOT NULL,
        result TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

# Add default user (admin / 1234) with hashed password
hashed_password = generate_password_hash("1234")
try:
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('admin', hashed_password))
    print("✅ Admin user created.")
except sqlite3.IntegrityError:
    print("⚠️ Admin user already exists.")

conn.commit()
conn.close()

print("✅ users.db is ready with users and history tables (username column included).")
