import tensorflow as tf
import numpy as np
import pickle
from keras.preprocessing import image
from flask import Flask, request, render_template, send_from_directory


# Create flask app
flask_app = Flask(__name__)

# Load model
model = pickle.load(open("Leaf Disease Detection Model.pkl", "rb"))

@flask_app.route("/")

def Home():
    return render_template("index.html")

@flask_app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@flask_app.route("/predict", methods = ["POST"])
def predict():

    image_uploaded = request.files['image']

    if image_uploaded.filename == '':
        return render_template('index.html', error="No Image File Uploaded")
    
    uploaded_image_path = f'uploads/{image_uploaded.filename}'
    image_uploaded.save(uploaded_image_path)

    test_image = image.load_img(uploaded_image_path, target_size = (128, 128), color_mode= 'rgb')
    test_image = image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)

    result = model.predict(test_image)
    index = np.argmax(result)

    for key, value in labels.items():
        if index == value:
            prediction = key


    mylist = prediction.split("___")
    leaf = mylist[0]
    disease = mylist[1]
    status = "Unhealthy"

    if disease == "Healthy":
        disease = "No Disease"
        status = "Healthy"

    print(f"Leaf : {leaf}")
    print(f"Status : {status}")
    print(f"Disease : {disease}")
    print(f"Probability : {np.round(result[0][index] * 100 , 2)}%")
    
    prediction_result = {"Leaf" : leaf , "Status" : status, "Disease" : disease }
    
    return render_template("index.html", uploaded_image = uploaded_image_path , prediction = prediction_result)


labels = {'Apple___Apple Scab': 0,
'Apple___Black Rot': 1,
'Apple___Cedar Apple Rust': 2,
'Apple___Healthy': 3,
'Bell Pepper___Bacterial Spot': 4,
'Bell Pepper___Healthy': 5,
'Blueberry___Healthy': 6,
'Cherry___Healthy': 7,
'Cherry___Powdery Mildew': 8,
'Corn (Maize)___Common Rust': 9,
'Corn (Maize)___Gray Leaf Spot (Cercospora Leaf Spot)': 10,
'Corn (Maize)___Healthy': 11,
'Corn (Maize)___Northern Leaf Blight': 12,
'Grape___Black Rot': 13,
'Grape___Esca (Black Measles)': 14,
'Grape___Healthy': 15,
'Grape___Leaf Blight (Isariopsis Leaf Spot)': 16,
'Orange___Citrus Greening (Huanglongbing)': 17,
'Peach___Bacterial Spot': 18,
'Peach___Healthy': 19,
'Potato___Early Blight': 20,
'Potato___Healthy': 21,
'Potato___Late_Blight': 22,
'Raspberry___Healthy': 23,
'Soybean___Healthy': 24,
'Squash___Powdery Mildew': 25,
'Strawberry___Healthy': 26,
'Strawberry___Leaf Scorch': 27,
'Tomato___Bacterial Spot': 28,
'Tomato___Early Blight': 29,
'Tomato___Healthy': 30,
'Tomato___Late Blight': 31,
'Tomato___Leaf Mold': 32,
'Tomato___Septoria Leaf Spot': 33,
'Tomato___Spider Mites (Two-Spotted Spider Mite)': 34,
'Tomato___Target Spot': 35,
'Tomato___Tomato Mosaic Virus': 36,
'Tomato___Tomato Yellow Leaf Curl Virus': 37}


if __name__ == "__main__":
    flask_app.run(host='0.0.0.0',port='8080')

