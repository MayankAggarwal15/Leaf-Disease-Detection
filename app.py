import tensorflow as tf
import numpy as np
# import pickle
from keras.preprocessing import image
from flask import Flask, request, render_template, send_from_directory


# Create flask app
flask_app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("Leaf Disease Detection Model.keras")
# model = pickle.load(open("Leaf Disease Detection Model.pkl", "rb"))

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


    leaf = labels[index]['leaf']
    disease = labels[index]['disease']
    status = labels[index]['status']
    treatment = labels[index]['treatment']
    precaution = labels[index]['precaution']



    print(f"Leaf : {leaf}")
    print(f"Status : {status}")
    print(f"Disease : {disease}")
    print(f"Treatment : {treatment}")
    print(f"Precaution : {precaution}")
    print(f"Probability : {np.round(result[0][index] * 100 , 2)}%")
    
    prediction_result = {"Leaf" : leaf , "Status" : status, "Disease" : disease, "Treatment" : treatment, "Precaution": precaution}
    
    return render_template("index.html", uploaded_image = uploaded_image_path , prediction = prediction_result)



labels = [
    { 
    'index' : 0,
    'leaf' : 'Apple',
    'disease' : 'Apple Scab',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice proper sanitation to control apple scab',
    'precaution' : 'Remove fallen leaves and fruit to reduce overwintering of fungal spores',
    },

    { 
    'index' : 1,
    'leaf' : 'Apple',
    'disease' : 'Black Rot',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and remove infected plant parts to manage black rot',
    'precaution' : 'Prune trees to improve air circulation and avoid overhead watering',
    },

    { 
    'index' : 2,
    'leaf' : 'Apple',
    'disease' : 'Cedar Apple Rust',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and remove alternate hosts to control cedar apple rust',
    'precaution' : 'Remove nearby juniper or cedar trees to reduce disease spread',
    },

    { 
    'index' : 3,
    'leaf' : 'Apple',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 4,
    'leaf' : 'Bell Pepper',
    'disease' : 'Bacterial Spot',
    'status' : 'Unhealthy',
    'treatment' : 'Apply copper-based fungicides and practice crop rotation to control bacterial spot',
    'precaution' : 'Avoid overhead irrigation and remove infected plant debris',
    },

    { 
    'index' : 5,
    'leaf' : 'Bell Pepper',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 6,
    'leaf' : 'Blueberry',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 7,
    'leaf' : 'Cherry',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 8,
    'leaf' : 'Cherry',
    'disease' : 'Powdery Mildew',
    'status' : 'Unhealthy',
    'treatment' : 'Apply sulfur or fungicides to manage powdery mildew',
    'precaution' : 'Plant resistant cherry varieties and ensure good air circulation',
    },

    { 
    'index' : 9,
    'leaf' : 'Corn (Maize)',
    'disease' : 'Common Rust',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and plant resistant corn varieties to control common rust',
    'precaution' : 'Remove corn debris and practice crop rotation',
    },

    { 
    'index' : 10,
    'leaf' : 'Corn (Maize)',
    'disease' : 'Gray Leaf Spot (Cercospora Leaf Spot)',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice crop rotation to manage gray leaf spot',
    'precaution' : 'Ensure good soil drainage and avoid overhead irrigation',
    },

    { 
    'index' : 11,
    'leaf' : 'Corn (Maize)',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 12,
    'leaf' : 'Corn (Maize)',
    'disease' : 'Northern Leaf Blight',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice crop rotation to control northern leaf blight',
    'precaution' : 'Remove corn debris and avoid planting susceptible hybrids',
    },

    { 
    'index' : 13,
    'leaf' : 'Grape',
    'disease' : 'Black Rot',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and remove infected plant parts to manage black rot',
    'precaution' : 'Prune grapevines to improve air circulation and reduce disease pressure',
    },

    { 
    'index' : 14,
    'leaf' : 'Grape',
    'disease' : 'Esca (Black Measles)',
    'status' : 'Unhealthy',
    'treatment' : 'Apply systemic fungicides and practice pruning to control esca',
    'precaution' : 'Avoid excessive pruning and maintain vineyard hygiene',
    },

    { 
    'index' : 15,
    'leaf' : 'Grape',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 16,
    'leaf' : 'Grape',
    'disease' : 'Leaf Blight (Isariopsis Leaf Spot)',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice canopy management to manage leaf blight',
    'precaution' : 'Ensure good air circulation and avoid overhead irrigation',
    },

    { 
    'index' : 17,
    'leaf' : 'Orange',
    'disease' : 'Citrus Greening (Huanglongbing)',
    'status' : 'Unhealthy',
    'treatment' : 'There is no cure for citrus greening; manage by removing infected trees and controlling the Asian citrus psyllid vector',
    'precaution' : 'Monitor for symptoms, implement vector control measures, and practice good orchard hygiene',
    },

    { 
    'index' : 18,
    'leaf' : 'Peach',
    'disease' : 'Bacterial Spot',
    'status' : 'Unhealthy',
    'treatment' : 'Apply copper-based fungicides and practice proper pruning to control bacterial spot',
    'precaution' : 'Prune peach trees to improve air circulation and avoid overhead watering',
    },

    { 
    'index' : 19,
    'leaf' : 'Peach',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 20,
    'leaf' : 'Potato',
    'disease' : 'Early Blight',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice crop rotation to manage early blight',
    'precaution' : 'Plant resistant potato varieties and avoid overhead irrigation',
    },

    { 
    'index' : 21,
    'leaf' : 'Potato',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 22,
    'leaf' : 'Potato',
    'disease' : 'Late_Blight',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice proper crop rotation to control late blight',
    'precaution' : 'Remove infected plant debris and avoid planting susceptible potato varieties',
    },

    { 
    'index' : 23,
    'leaf' : 'Raspberry',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 24,
    'leaf' : 'Soybean',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 25,
    'leaf' : 'Squash',
    'disease' : 'Powdery Mildew ',
    'status' : 'Unhealthy',
    'treatment' : 'Apply sulfur or fungicides to manage powdery mildew',
    'precaution' : 'Plant resistant squash varieties and ensure good air circulation',
    },

    { 
    'index' : 26,
    'leaf' : 'Strawberry',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 27,
    'leaf' : 'Strawberry',
    'disease' : 'Leaf Scorch',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice proper irrigation management to control leaf scorch',
    'precaution' : 'Avoid overhead irrigation and remove infected plant debris',
    },

    { 
    'index' : 28,
    'leaf' : 'Tomato',
    'disease' : 'Bacterial Spot',
    'status' : 'Unhealthy',
    'treatment' : 'Apply copper-based fungicides and practice proper pruning to control bacterial spot',
    'precaution' : 'Prune tomato plants to improve air circulation and avoid overhead watering.',
    },

    { 
    'index' : 29,
    'leaf' : 'Tomato',
    'disease' : 'Early Blight',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice crop rotation to manage early blight.',
    'precaution' : 'Plant resistant tomato varieties and avoid overhead irrigation',
    },

    { 
    'index' : 30,
    'leaf' : 'Tomato',
    'disease' : 'No Disease',
    'status' : 'Healthy',
    'treatment' : 'No Treatment Required',
    'precaution' : 'No Precaution Required',
    },

    { 
    'index' : 31,
    'leaf' : 'Tomato',
    'disease' : 'Late Blight',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice proper crop rotation to control late blight',
    'precaution' : 'R,emove infected plant debris and avoid planting susceptible tomato varieties.',
    },

    { 
    'index' : 32,
    'leaf' : 'Tomato',
    'disease' : 'Leaf Mold',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice proper irrigation management to manage leaf mold',
    'precaution' : 'Avoid overhead irrigation and ensure good air circulation around tomato plants',
    },

    { 
    'index' : 33,
    'leaf' : 'Tomato',
    'disease' : 'Septoria Leaf Spot',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice crop rotation to control septoria leaf spot',
    'precaution' : 'Remove infected plant debris and avoid overhead irrigation',
    },

    { 
    'index' : 34,
    'leaf' : 'Tomato',
    'disease' : 'Spider Mites (Two-Spotted Spider Mite)',
    'status' : 'Unhealthy',
    'treatment' : 'Apply miticides and practice cultural controls to manage spider mites',
    'precaution' : 'Monitor for early signs of infestation and maintain proper plant hygiene.',
    },

    { 
    'index' : 35,
    'leaf' : 'Tomato',
    'disease' : 'Target Spot',
    'status' : 'Unhealthy',
    'treatment' : 'Apply fungicides and practice proper plant spacing to manage target spot',
    'precaution' : 'Remove infected plant debris and avoid overhead irrigation.',
    },

    { 
    'index' : 36,
    'leaf' : 'Tomato',
    'disease' : 'Tomato Mosaic Virus',
    'status' : 'Unhealthy',
    'treatment' : 'There is no cure for tomato mosaic virus; manage by removing infected plants and controlling aphid vectors',
    'precaution' : 'Monitor for symptoms, implement aphid control measures, and practice good sanitation',
    },

    { 
    'index' : 37,
    'leaf' : 'Tomato',
    'disease' : 'Tomato Yellow Leaf Curl Virus',
    'status' : 'Unhealthy',
    'treatment' : 'There is no cure for tomato yellow leaf curl virus; manage by removing infected plants and controlling whitefly vectors',
    'precaution' : 'Implement whitefly control measures, plant resistant tomato varieties, and practice good field sanitation',
    },
]



if __name__ == "__main__":
    flask_app.run(debug=True , port= 3000)

