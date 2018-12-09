FROM python:3.6
ENV MODELPATH /runtime

RUN pip3 install --upgrade pip
RUN mkdir $MODELPATH
WORKDIR $MODELPATH

ADD requirements.txt $MODELPATH
RUN pip3 install -r requirements.txt

ADD models $MODELPATH/models
ADD sentiment_predict.py $MODELPATH
RUN ls -ahl

CMD python3.6 sentiment_predict.py