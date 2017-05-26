FROM ubuntu

MAINTAINER Mehmet Alp Albyraktaroğlu

RUN apt-get update -y && apt-get install git python python-pip -y

RUN apt-get install python-tk -y

RUN pip install --upgrade pip

WORKDIR /code

ADD . /code

RUN pip install -r /code/requirements.txt
    
EXPOSE 9999

EXPOSE 80

CMD ["python", "server.py"]
