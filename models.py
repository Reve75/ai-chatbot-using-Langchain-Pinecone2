from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class QueryLog(Base):
    __tablename__ = 'query_logs'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    query = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
