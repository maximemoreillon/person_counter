FROM python:latest
WORKDIR /usr/src/app
COPY . .
RUN pip3 install -r requirements.txt
EXPOSE 8051
CMD [ "python3", "server.py" ]
