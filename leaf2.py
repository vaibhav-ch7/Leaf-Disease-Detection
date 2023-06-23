#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.applications.vgg19 import preprocess_input
filepath = 'C:/Users/HP/Desktop/leaf disease detection/best_model (1).h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")


def prediction(path):
  img = load_img(path,target_size = (256,256))
  i= img_to_array(img)
  #cv2.imwrite('Image_from_array.jpg',i)
  im = preprocess_input(i)
  print("\n\ndimensions of preprocessed array : ",i.shape)
  #cv2.imwrite('preprocess.jpg', im)
  img = np.expand_dims(im,axis=0)
  print("\n\nDiamensions of Expanded Array: ",img.shape)
  x=model.predict(img)
  print("\n\n",x)
  pred =np.argmax(x)
  print("\n\n",pred)
  
  if pred==0:
      return "Apple - Apple Scab", 'Apple___Apple_scab (1).html'
       
  elif pred==1:
      return "Apple - Black Rot", 'Apple___Black_rot (1).html'
        
  elif pred==2:
      return "Apple - Cedar Apple Rust", 'Apple___Cedar_apple_rust (1).html'
        
  elif pred==3:
      return "Apple - Healthy", 'Apple___healthy (1).html'
       
  elif pred==4:
      return "Blueberry - Healthy", 'Blueberry___healthy (1).html'
        
  elif pred==5:
      return "Cherry - Powdery Mildew", 'Cherry_(including_sour)___Powdery_mildew (1).html'
        
  elif pred==6:
      return "Cherry - Healthy", 'Cherry_(including_sour)__healthy (1).html'
        
  elif pred==7:
      return "Corn - Cercospora Gray Leaf Spot", 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot (1).html'

  elif pred==8:
      return "Corn - Common Rust", 'Corn_(maize)___Common_rust_ (1).html'
        
  elif pred==9:
      return "Corn - Nothern Leaf Blight", 'Corn_(maize)___Northern_Leaf_Blight (1).html'

  elif pred==10:
      return "Corn - Healthy", 'Corn_(maize)___healthy (1).html'
       
  elif pred==11:
      return "Grape - Black Rot", 'Grape___Black_rot (1).html'
        
  elif pred==12:
      return "Grape - Esca (Black Mealses)", 'Grape___Esca_(Black_Measles) (1).html'
        
  elif pred==13:
      return "Grape - Leaf Blight (Isariopsis Leaf Spot)", 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot) (1).html'
       
  elif pred==14:
      return "Grape - Healthy", 'Grape___healthy (1).html'
        
  elif pred==15:
      return "Orange - Citrus Greening (Haunlongbing)", 'Orange___Haunglongbing_(Citrus_greening) (1).html'
        
  elif pred==16:
      return "Peach - Bactorial Spot", 'Peach___Bacterial_spot (1).html'
        
  elif pred==17:
      return "Peach - Healthy", 'Peach___healthy (1).html'

  elif pred==18:
      return "Pepper Bell - Bactorial Spot", 'Pepper,_bell___Bacterial_spot (1).html'
        
  elif pred==19:
      return "Pepper Bell - Healthy", 'Pepper,_bell___healthy (1).html'

  elif pred==20:
      return "Potato - Early Blight", 'Potato___Early_blight (1).html'
       
  elif pred==21:
      return "Potato - Late Blight", 'Potato___Late_blight (1).html'
        
  elif pred==22:
      return "Potato - Healthy", 'Potato___healthy (1).html'
        
  elif pred==23:
      return "Raspberry - Healthy", 'Raspberry___healthy (1).html'
       
  elif pred==24:
      return "Soyabean - Healthy", 'Soybean___healthy (1).html'
        
  elif pred==25:
      return "Squash - Powdery Mildew", 'Squash___Powdery_mildew (1).html'
        
  elif pred==26:
      return "Strawberry - Leaf Scorch", 'Strawberry___Leaf_scorch (1).html'
        
  elif pred==27:
      return "Strawberry - Healthy", 'Strawberry___healthy (1).html'

  elif pred==28:
      return "Tomato - Bactorial Spot", 'Tomato___Bacterial_spot (1).html'
        
  elif pred==29:
      return "Tomato - Early Blight", 'Tomato___Early_blight (1).html'

  elif pred==30:
      return "Tomato - Late Blight", 'Tomato___Late_blight (1).html'
       
  elif pred==31:
      return "Tomato - Leaf Mold", 'Tomato___Leaf_Mold (1).html'
        
  elif pred==32:
      return "Tomato - Septorial Leaf Spot", 'Tomato___Septoria_leaf_spot (1).html'
        
  elif pred==33:
      return "Tomato - Two Spotted Spider Mites", 'Tomato___Spider_mites Two-spotted_spider_mite (1).html'
       
  elif pred==34:
      return "Tomato - Target Spot", 'Tomato___Target_Spot (1).html'
        
  elif pred==35:
      return "Tomato - Tomoato Yellow Leaf Curl Virus Disease", 'Tomato___Tomato_Yellow_Leaf_Curl_Virus (1).html'
        
  elif pred==36:
      return "Tomato - Mosaic Virus Disease", 'Tomato___Tomato_mosaic_virus (1).html'
        
  elif pred==37:
      return "Tomato - Healthy", 'Tomato___healthy (1).html'
  
  
    

# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("Posted Input = ", filename)
        
        file_path = os.path.join('C:/Users/HP/Desktop/leaf disease detection/original dataset/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = prediction(path=file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,debug='True',port=4000)
    
    
