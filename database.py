from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# SQLite database URL for local development
DATABASE_URL = "sqlite:///./news.db"  # Use triple slashes for SQLite local file DB
# Create the engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal is used to get a database session object
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


#Base.metadata.create_all(bind=engine)