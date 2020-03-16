FROM python:3
WORKDIR /usr/src/app
COPY . .
RUN python --version
RUN pip install -r requirements.txt
EXPOSE 8051
CMD [ "python", "server.py" ]
