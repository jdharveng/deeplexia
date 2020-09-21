FROM python:3.6

ADD ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN gdown https://drive.google.com/uc?id=1NiEvwVxu9eDYF7m5AdEZV0gopaqRB-7u&export=download
ADD * ./

ENTRYPOINT ["python", "server_deeplexia.py"]

