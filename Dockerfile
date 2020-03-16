# Tensorflow only works with Python 3.6.X
FROM python:3.6
WORKDIR /usr/src/app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8051
CMD [ "echo", "'start!'" ]
CMD [ "python", "server.py" ]
