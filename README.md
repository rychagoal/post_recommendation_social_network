# Recommendation System for Posts in Social Network

Hello, everyone! This project is focusing on machine learning and APIs. I hope it proves useful for those who are still learning about ML, and want to understand how ML can be implemented to solve real-world problems. This project serves as an example of how to achieve this.

## Project Overview

The project involves an API designed to recommend posts to users in a social network. Note that this project does not include a graphical user interface (GUI); instead, it focuses on backend development. The results can be viewed in JSON format via your browser or Postman. The code also includes database connectivity to handle data retrieval and submission.
I did not focus on exploratory data analysis (EDA) because it largely depends on your specific dataset. However, I have included some Jupyter notebooks demonstrating how to prepare data before feeding it into the model and how to select appropriate features.

## Project Purpose

We are all users of social networks and want to read relevant information and topics that interest us. The purpose of this model is to recommend the top 5 most relevant posts for each user based on the personal features we know about them.
Feel free to explore the code, and let me know if you have any questions or feedback!

> ## Useful instruments to use in this project:
>
> - Model: CatBoostClassifier
> - EDA: python, pandas, numpy, mathplotlib, seaborn, scikit-learn
> - Feature engeneering: pytorch, transformers
> - Service: FastApi, SQLAlchemy

![PANDAS](https://img.shields.io/badge/PANDAS-1.4.2-090909??style=flat-square&logo=PANDAS) ![NUMPY](https://img.shields.io/badge/NUMPY-1.22.4-090909??style=flat-square&logo=NUMPY) ![fastapi](https://img.shields.io/badge/FASTAPI-0.75.1-090909??style=flat-square&logo=fastapi) ![sqlalchemy](https://img.shields.io/badge/SQLALCHEMY-1.4.35-090909??style=flat-square&logo=sqlalchemy) ![catboost](https://img.shields.io/badge/CATBOOST-1.0.6-090909??style=flat-square&logo=catboost) ![pydantic](https://img.shields.io/badge/PYDANTIC-1.9.1-090909??style=flat-square&logo=pydantic) ![psycopg2](https://img.shields.io/badge/PSYCOPG2-2.9.3-090909??style=flat-square&logo=psycopg2) ![uvicorn](https://img.shields.io/badge/UVICORN-0.16.0-090909??style=flat-square&logo=uvicorn)

## Uploaded files:
1. [app.py][df1] - The service that downloads data, loads the model and issues recommendations in JSON format
2. [EDA.ipynb][df2] - Notebook with brief exploratory data analysis
3. [load_data_final.ipynb][df3] - Separate notebook with how to load final data
4. [model_training.ipynb][df4] - Notebook with model training
5. [model][df5] - Pretrained model using Catboost

[df1]: <app.py>
[df2]: <EDA.ipynb>
[df3]: <load_data_final.ipynb>
[df4]: <model_training.ipynb>
[df5]: <model>
