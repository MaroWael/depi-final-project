import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager

# Database configuration
DB_CONFIG = {
    "host": "serverless-eu-west-3.sysp0000.db1.skysql.com",
    "port": 4027,
    "user": "dbpwf06347787",
    "password": "e3oQXYe]tCUWOpLGA0nGST",
    "database": "ai_db"
}



def get_db_connection():
    """Create and return a database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise

@contextmanager
def get_db_cursor(dictionary=True):
    """Context manager for database operations"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=dictionary)
    try:
        yield cursor
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()

def init_database():
    """Initialize database tables"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                image_url TEXT,
                role ENUM('admin', 'chief') NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create login_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS login_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR(45),
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Create video_reports table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_reports (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_filename VARCHAR(255) NOT NULL,
                report_data JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        connection.commit()
        print("Database tables created successfully")
    except Error as e:
        print(f"Error creating tables: {e}")
        raise
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    init_database()
