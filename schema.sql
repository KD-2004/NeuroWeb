-- Drop tables if they exist to allow re-initialization (use with caution in production)
-- Dropping tables in reverse order of foreign key dependencies
DROP TABLE IF EXISTS learning_interactions;
DROP TABLE IF EXISTS learning_profiles;
DROP TABLE IF EXISTS password_reset_tokens;
DROP TABLE IF EXISTS chat_messages; -- Drop chat_messages last as it might reference users

DROP TABLE IF EXISTS users; -- Drop users table


-- Create the users table
-- Based on mia.py and your second users table definition
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL, -- Using email as required by mia.py auth routes
    password_hash TEXT NOT NULL,
    registered_on DATETIME DEFAULT CURRENT_TIMESTAMP,
    email_verified BOOLEAN DEFAULT FALSE,
    email_verified_on DATETIME,
    is_admin BOOLEAN DEFAULT FALSE, -- Optional: if you need admin users
    last_login DATETIME -- Added last_login based on mia.py
);

-- Create the learning_profiles table
-- Based on your definition
CREATE TABLE IF NOT EXISTS learning_profiles (
    user_id INTEGER PRIMARY KEY,
    learning_style TEXT CHECK(learning_style IN ('visual', 'auditory', 'kinesthetic', 'mixed')) DEFAULT 'mixed',
    difficulty_level INTEGER CHECK(difficulty_level BETWEEN 1 AND 10) DEFAULT 5,
    preferred_topics TEXT,  -- JSON array (consider storing as JSON or a separate table if complex)
    weak_topics TEXT,       -- JSON array (consider storing as JSON or a separate table if complex)
    strengths TEXT,         -- JSON object {topic: strength_score} (consider storing as JSON or a separate table if complex)
    last_active TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE -- Add ON DELETE CASCADE for profiles
);

-- Create the learning_interactions table
-- Based on your definition
CREATE TABLE IF NOT EXISTS learning_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    interaction_type TEXT CHECK(interaction_type IN ('chat', 'lesson', 'quiz', 'practice', 'review')),
    topic TEXT NOT NULL,
    content TEXT NOT NULL,  -- JSON structure varies by type (consider storing as JSON or a separate table)
    performance_metrics TEXT, -- JSON object (consider storing as JSON or a separate table)
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE -- Add ON DELETE CASCADE for interactions
);

-- Create the password reset tokens table
-- Based on your second definition and mia.py
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token TEXT UNIQUE NOT NULL,
    expires_at DATETIME NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE -- Add ON DELETE CASCADE
);

-- Create the chat_messages table
-- Based on your first definition, including nullable user_id and session_id
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, -- Can be JWT token JTI or a separate session ID if unauth chat is needed
    user_id INTEGER,          -- Nullable if chat can be anonymous before login
    role TEXT NOT NULL,       -- 'User', 'AI', 'System', 'Error'
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL -- Set user_id to NULL if user is deleted
);
