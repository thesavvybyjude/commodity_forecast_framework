"""
P5-T6: Subscriber Manager
SQLite-based subscriber list for price alerts.
"""
import logging
import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "subscribers.db"


class SubscriberManager:
    """Manages email subscribers for price spike alerts."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        """Create the subscribers table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subscribers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    commodity TEXT NOT NULL,
                    active INTEGER DEFAULT 1
                )
            """)
            conn.commit()

    def add(self, email: str, commodity: str):
        """Add a new subscriber for a commodity."""
        with sqlite3.connect(self.db_path) as conn:
            # Check for existing subscription
            cursor = conn.execute(
                "SELECT id FROM subscribers WHERE email = ? AND commodity = ?",
                (email, commodity),
            )
            if cursor.fetchone():
                logger.info("Subscriber %s already exists for %s", email, commodity)
                return

            conn.execute(
                "INSERT INTO subscribers (email, commodity) VALUES (?, ?)",
                (email, commodity),
            )
            conn.commit()
            logger.info("Added subscriber %s for %s", email, commodity)

    def get_subscribers(self, commodity: str) -> list:
        """Return list of active emails for a given commodity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT email FROM subscribers WHERE commodity = ? AND active = 1",
                (commodity,),
            )
            return [row[0] for row in cursor.fetchall()]

    def remove(self, email: str, commodity: str = None):
        """Deactivate a subscriber."""
        with sqlite3.connect(self.db_path) as conn:
            if commodity:
                conn.execute(
                    "UPDATE subscribers SET active = 0 WHERE email = ? AND commodity = ?",
                    (email, commodity),
                )
            else:
                conn.execute(
                    "UPDATE subscribers SET active = 0 WHERE email = ?",
                    (email,),
                )
            conn.commit()
            logger.info("Deactivated subscriber %s", email)

    def list_all(self) -> list:
        """Return all active subscriptions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT email, commodity FROM subscribers WHERE active = 1"
            )
            return [{'email': row[0], 'commodity': row[1]} for row in cursor.fetchall()]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sm = SubscriberManager()
    print("Current subscribers:")
    for sub in sm.list_all():
        print(f"  {sub['email']} -> {sub['commodity']}")
