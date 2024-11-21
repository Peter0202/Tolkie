FROM python:3.9
WORKDIR /app

COPY app.py /app/app.py
COPY prompts /app/prompts

RUN pip install flask
RUN pip install transformers 
RUN pip install requests

EXPOSE 5000

CMD ["python", "app.py"]
