import requests
import psycopg2
import pytest

def test_mlflow_reachable():
    resp = requests.get("http://localhost:5000/health")
    assert resp.status_code == 200

def test_postgres_connection():
    conn = psycopg2.connect(
        host="localhost", port=5432,
        dbname="mlflow", user="mlflow", password="mlflow"
    )
    assert conn.status == 1  # CONNECTION_OK
    conn.close()