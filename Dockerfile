FROM python:3.12
WORKDIR /opt/source-code/
COPY . /opt/source-code/
RUN pip install -r requirements.txt
RUN python main.py