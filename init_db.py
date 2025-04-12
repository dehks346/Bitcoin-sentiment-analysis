from database import Base, engine
from models import NewsArticle, TrainingData

# Create all tables defined in models.py
Base.metadata.create_all(bind=engine)
print("Database tables created successfully.")