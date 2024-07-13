import os
import pandas as pd
from typing import List
from fastapi import FastAPI, Depends
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, String, Integer, desc
from pydantic import BaseModel

# Connect database first
SQLALCHEMY_DATABASE_URL = "postgresql://user:password@host:dbname"
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

app = FastAPI()

#Structure the response data with Pydantic
class Post(Base):
    __tablename__ = "post"
    __table_args__ = {"schema": "public"}
    id = Column(Integer, primary_key = True)
    text = Column(String) 
    topic = Column(String) 
  
class PostGet(BaseModel):
    id: int
    text: str
    topic: str 
    
    class Config:
        orm_mode = True
        
#manage database sessions within API requests
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#load model
def get_model_path(path: str) -> str:
    MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("~/model.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model
    
#load user features    
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://user:password@host:dbname", pool_size=10, max_overflow=20
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features() -> pd.DataFrame:
    '''load user features prepared beforehand'''
    new_feature = """SELECT * FROM user_features"""
    loaded_data = batch_load_sql(new_feature)
    
    return loaded_data
    
def load_posts() -> pd.DataFrame:
    '''load data of posts'''
    loaded_data2 = batch_load_sql('SELECT * FROM post')
    
    return loaded_data2
        
def compose_user_posts(user_id, timestamp):
    '''make data of one user and all the posts'''
    df_table = pd.concat([df_users[df_users['user_id'] == user_id], df_posts], axis=1).reset_index().drop('index', axis=1)

    for column in df_users.drop('index', axis=1).columns:
            df_table.loc[df_table[column].isna(), column] = df_table.loc[0, column]

    df_table = df_table.drop(['level_0'], axis=1)

    df_table['timestamp'] = pd.Timestamp(timestamp)
    df_table['timestamp'] = df_table['timestamp'].astype('int64')

    return df_table
        
def predict_posts(model: CatBoostClassifier, id: int, time: datetime, top_n: int = 5) -> List[int]:
    '''make top 5 predictions using the model'''
    features = ['timestamp', 'user_id', 'post_id', 'gender', 'age', 'country', 'city', 'exp_group', 'os', 'source', 'Mean_TFIDF', 'pca_0', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10', 'pca_11', 'pca_12', 'pca_13', 'pca_14']
    
    user_features = compose_user_posts(id, time)
    user_features = user_features[features]
    user_features = user_features[user_features['user_id'] == id]

    # Handle NaN values in categorical columns by converting them to strings
    cat_cols = ['country', 'city', 'source', 'os']
    for col in cat_cols:
        user_features[col] = user_features[col].fillna('NaN').astype(str)
        
    #define categorical features
    cat_features = [user_features.columns.get_loc(col) for col in cat_cols]
    
    # Predictions
    predict_pool = Pool(user_features, cat_features=cat_features)
    predictions = model.predict_proba(predict_pool)[:, 1]
    
    # Add predictions back to user_features DataFrame
    user_features['predictions'] = predictions
    
    # Sort by predictions to get top N posts
    top_posts = user_features.sort_values(by='predictions', ascending=False).head(top_n)
    
    # Assuming 'post_id' is in user_features to identify top posts
    top_post_ids = top_posts['post_id'].tolist()
    
    return top_post_ids
    
    
model = load_models()
df_users = load_features()
df_posts = load_posts()

#FastAPI endpoint for top 5 predicted posts
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5, db: Session = Depends(get_db)) -> List[PostGet]:
    post_ids = predict_posts(model, id, time, top_n=limit)
    posts = []
    if post_ids:
        for post_id in post_ids:
            post = db.query(Post).filter(Post.id == post_id).first()
            if post is not None:
                post_get = PostGet(id=post.id, text=post.text, topic=post.topic)
                posts.append(post_get)
    return posts
