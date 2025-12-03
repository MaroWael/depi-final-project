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
    
    @staticmethod
    def update(user_id: int, name: str = None, email: str = None, password: str = None, image_url: str = None, role: str = None):
        """Update user information"""
        with get_db_cursor() as cursor:
            # Build dynamic update query
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if email is not None:
                updates.append("email = %s")
                params.append(email)
            if password is not None:
                updates.append("password = %s")
                params.append(password)
            if image_url is not None:
                updates.append("image_url = %s")
                params.append(image_url)
            if role is not None:
                updates.append("role = %s")
                params.append(role)
            
            if not updates:
                return False
            
            params.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
            cursor.execute(query, params)
            return cursor.rowcount > 0

    @staticmethod
    def delete(user_id: int):
        """Delete a user"""
        with get_db_cursor() as cursor:
            query = "DELETE FROM users WHERE id = %s"
            cursor.execute(query, (user_id,))
            return cursor.rowcount > 0
    
    @staticmethod
    def email_exists_excluding_user(email: str, user_id: int) -> bool:
        """Check if email exists for other users (used during update)"""
        with get_db_cursor() as cursor:
            query = "SELECT COUNT(*) as count FROM users WHERE email = %s AND id != %s"
            cursor.execute(query, (email, user_id))
            result = cursor.fetchone()
            return result['count'] > 0

class VideoReport:
    @staticmethod
    def create(video_filename: str, report_data: str):
        """Create a new video report"""
        with get_db_cursor() as cursor:
            query = """
                INSERT INTO video_reports (video_filename, report_data)
                VALUES (%s, %s)
            """
            cursor.execute(query, (video_filename, report_data))
            return cursor.lastrowid

    @staticmethod
    def get_all():
        """Get all video reports"""
        with get_db_cursor() as cursor:
            query = "SELECT * FROM video_reports ORDER BY created_at DESC"
            cursor.execute(query)
            return cursor.fetchall()

    @staticmethod
    def get_by_id(report_id: int):
        """Get video report by ID"""
        with get_db_cursor() as cursor:
            query = "SELECT * FROM video_reports WHERE id = %s"
            cursor.execute(query, (report_id,))
            return cursor.fetchone()


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
