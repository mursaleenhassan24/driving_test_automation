import sqlite3


class Database:
    def __init__(self):
        self.conn = sqlite3.connect('database.db')
        self.cursor = self.conn.cursor()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            coords TEXT,
            distance REAL,
            speed REAL
            )
        ''')
        self.conn.commit()

    def insert(self, track_id, coords, distance, speed):
        self.cursor.execute('''
            INSERT INTO data (track_id, coords, distance, speed)
            VALUES (?, ?, ?, ?)
        ''', (track_id, coords, distance, speed))  # Supply all four parameters here
        self.conn.commit()

    def delete_all(self):
        self.cursor.execute('DELETE FROM data')
        self.conn.commit()

    def select(self):
        self.cursor.execute('SELECT * FROM data')
        return self.cursor.fetchone()

