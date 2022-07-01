import json
import celery as cel
# from celery import Celery
import os

broker = os.getenv('BROKER', 'redis://localhost:6379/0')
app = cel.Celery('risk_barra_model', broker=broker,
            backend=broker,
            include=['risk_barra_model.portfolio_analysis'])

if __name__ == '__main__':
    app.start()