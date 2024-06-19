FROM python:3.9

WORKDIR /code

ENV HF_HOME=/code/.cache

COPY ./requirements.txt /code/requirements.txt
COPY ./main.py /code/main.py
COPY ./tinyllama-1.1b-chat.gguf ./tinyllama-1.1b-chat.gguf

RUN mkdir -p /code/.cache && chmod -R 777 /code/.cache

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
