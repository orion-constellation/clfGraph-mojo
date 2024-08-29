from datetime import datetime
from uuid import uuid4

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Set up SQLAlchemy
DATABASE_URL = "postgresql+psycopg2://your_user:your_password@localhost/sklearn_project"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define models
class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    datasets = relationship("Dataset", back_populates="session")
    models = relationship("Model", back_populates="session")

class Dataset(Base):
    __tablename__ = "datasets"
    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    dataset_name = Column(String, nullable=False)
    data_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    session = relationship("Session", back_populates="datasets")

class Model(Base):
    __tablename__ = "models"
    model_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_params = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    session = relationship("Session", back_populates="models")
    results = relationship("Result", back_populates="model")

class Result(Base):
    __tablename__ = "results"
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.model_id"), nullable=False)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    model = relationship("Model", back_populates="results")

# Create tables
Base.metadata.create_all(bind=engine)

# Database operations
def create_session(session_name: str):
    db = SessionLocal()
    new_session = Session(session_name=session_name)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    db.close()
    return new_session.session_id

def add_dataset(session_id: str, dataset_name: str, data_path: str):
    db = SessionLocal()
    new_dataset = Dataset(session_id=session_id, dataset_name=dataset_name, data_path=data_path)
    db.add(new_dataset)
    db.commit()
    db.close()

def add_model(session_id: str, model_name: str, model_type: str, model_params: dict):
    db = SessionLocal()
    new_model = Model(session_id=session_id, model_name=model_name, model_type=model_type, model_params=model_params)
    db.add(new_model)
    db.commit()
    db.close()

def add_results(model_id: int, session_id: str, accuracy: float, precision: float, recall: float, f1_score: float):
    db = SessionLocal()
    new_result = Result(model_id=model_id, session_id=session_id, accuracy=accuracy, precision=precision, recall=recall, f1_score=f1_score)
    db.add(new_result)
    db.commit()
    db.close()

def fetch_sessions():
    db = SessionLocal()
    sessions = db.query(Session).all()
    db.close()
    return sessions

def fetch_models_for_session(session_id: str):
    db = SessionLocal()
    models = db.query(Model).filter(Model.session_id == session_id).all()
    db.close()
    return models

# Example usage
if __name__ == "__main__":
    session_id = create_session("My First SQLAlchemy Session")
    add_dataset(session_id, "Iris Dataset", "/path/to/iris.csv")
    add_model(session_id, "Random Forest Classifier", "RandomForest", {"n_estimators": 100, "max_depth": 5})
    models = fetch_models_for_session(session_id)
    print(models)