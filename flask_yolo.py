from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/api/object_detection', methods=['POST'])
def DetectImage():
    imageFile = []
    imageFromClient = request.files['image'].read()
    # creating network and putting names in imageFile
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    layerName = net.getLayerNames()
    output_layers = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    with open("coco.names", "r") as f:
        imageFile = [line.strip() for line in f.readlines()]

    np_buffer = np.frombuffer(imageFromClient, np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    image = cv2.resize(image, None, fx=0.4, fy=0.4)
    imageBlob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, False)
    net.setInput(imageBlob)
    detect = net.forward(output_layers)
    objectDict = {}
    for objects in detect:
        for detection in objects:
            scores = detection[5:]
            classId = np.argmax(scores)
            accuracy = scores[classId]
            if accuracy >= 0.1:
                if str(imageFile[classId]) not in objectDict:
                    objectDict[str(imageFile[classId])] = float(accuracy) * 100
                else:
                    if objectDict[str(imageFile[classId])] < float(accuracy) * 100:
                        objectDict[str(imageFile[classId])] < float(accuracy) * 100
    objectList = []
    for labelImage, accuracyImage in objectDict.items():
        objectList.append({'label': labelImage, 'accuracy': accuracyImage})

    response = jsonify({"objects": objectList})
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True)
