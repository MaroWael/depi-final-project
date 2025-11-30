from database import get_db_cursor

class User:
    @staticmethod
    def create(name: str, email: str, password: str, image_url: str, role: str):
        """Create a new user"""
        with get_db_cursor() as cursor:
            query = """
                INSERT INTO users (name, email, password, image_url, role)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (name, email, password, image_url, role))
            return cursor.lastrowid
    
    @staticmethod
    def get_by_email(email: str):
        """Get user by email"""
        with get_db_cursor() as cursor:
            query = "SELECT * FROM users WHERE email = %s"
            cursor.execute(query, (email,))
            return cursor.fetchone()
    
    @staticmethod
    def get_by_id(user_id: int):
        """Get user by ID"""
        with get_db_cursor() as cursor:
            query = "SELECT * FROM users WHERE id = %s"
            cursor.execute(query, (user_id,))
            return cursor.fetchone()
    
    @staticmethod
    def email_exists(email: str) -> bool:
        """Check if email already exists"""
        with get_db_cursor() as cursor:
            query = "SELECT COUNT(*) as count FROM users WHERE email = %s"
            cursor.execute(query, (email,))
            result = cursor.fetchone()
            return result['count'] > 0
    
    @staticmethod
    def get_all_chiefs():
        """Get all users with chief role"""
        with get_db_cursor() as cursor:
            query = "SELECT id, name, email, image_url, role, created_at FROM users WHERE role = 'chief'"
            cursor.execute(query)
            return cursor.fetchall()


class LoginHistory:
    @staticmethod
    def create(user_id: int, ip_address: str = None, user_agent: str = None):
        """Record a login event"""
        with get_db_cursor() as cursor:
            query = """
                INSERT INTO login_history (user_id, ip_address, user_agent)
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (user_id, ip_address, user_agent))
            return cursor.lastrowid
    
    @staticmethod
    def get_user_history(user_id: int, limit: int = 10):
        """Get login history for a user"""
        with get_db_cursor() as cursor:
            query = """
                SELECT * FROM login_history 
                WHERE user_id = %s 
                ORDER BY login_time DESC 
                LIMIT %s
            """
            cursor.execute(query, (user_id, limit))
            return cursor.fetchall()
