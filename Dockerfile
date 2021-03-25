FROM python:3
RUN mkdir /app/
COPY ["./flask_yolo.py", "./coco.names", "./requirements.txt","./yolov3-tiny.cfg","./yolov3-tiny.weights","./app/"]
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python","flask_yolo.py"]
EXPOSE 5000