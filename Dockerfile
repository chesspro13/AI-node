FROM python:3.13

WORKDIR /

ADD api.py .
ADD requirements.txt .

RUN pip install -r requirements.txt

ENV AM_I_IN_A_DOCKER_CONTAINER=True

CMD ["uvicorn", "--host", "0.0.0.0", "api:api"]