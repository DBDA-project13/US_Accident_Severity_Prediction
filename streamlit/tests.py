from db_mysql_config import AccidentPredictionDB

db = AccidentPredictionDB()
if db.connect():
    print("✅ Connected!")
    db.create_tables()
    print("✅ Tables created!")
else:
    print("❌ Connection failed!")