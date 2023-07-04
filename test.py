from flask import Flask, request, abort, make_response
from nsfw_detector import predict

app = Flask(__name__)
model = predict.load_model('nsfw_mobilenet2.224x224.h5')
#model = predict.load_model('static/models/nsfw_mobilenet2.224x224.h5')

# NSFW
@app.route('/nsfw-api', methods=['POST', 'GET'])
def check_nsfw():
    img_url = request.args.get('url')
    images_url = [img_url]

    return predict.classify(model, images_url)[img_url]
