FROM python:latest
WORKDIR /usr/src/app
COPY . .
RUN pip3 install
EXPOSE 8051
CMD [ "python3", "server.py" ]
