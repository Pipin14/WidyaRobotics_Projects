FROM bitnami/pytorch

ENV PORT=8000

WORKDIR /app

COPY . /app
# Login to user 'root'
USER root

RUN touch database.json
# Give permission to write database.json
RUN chmod +w database.json 

RUN pip install -r requirements.txt --default-timeout=3600

EXPOSE $PORT

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT