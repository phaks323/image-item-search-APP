import numpy as np
import os
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image
import keras.utils as image
import time
time.clock = time.time
tic = time.clock()
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
import random
import pickle
model = keras.applications.VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)


menu = ["Picture","Webcam"]
choice = st.sidebar.selectbox("Input type",menu)
#Put slide to adjust tolerance

#Infomation section 
st.sidebar.title("Student Information")

if choice == "Picture":
    st.title("Face Recognition App")
    image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])

elif choice == "Webcam":
    st.title("Face Recognition App")
    #Camera Settings
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    FRAME_WINDOW = st.image([])
    
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            st.info("Please turn off the other app that is using the camera and restart app")
            st.stop()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        FRAME_WINDOW.image(image)

file1 = open("images.p",'rb')
images_t = pickle.load(file1)
file1.close()

file2 = open("pca_features.p",'rb')
pca_features_t = pickle.load(file2)
file2.close()

file3 = open("pca.p",'rb')
pca_t = pickle.load(file3)
file3.close()

file4 = open("feat_extractor.p",'rb')
feat_extractor_t = pickle.load(file4)
file4.close()

# images_path = 'archive/images'
# image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)
# max_num_images = 1000

# images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
# if max_num_images < len(images):
#     images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]
# print(images)
# print("keeping %d images to analyze" % len(images))

def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


# features = []
# for i, image_path in enumerate(images):
#     if i % 500 == 0:
#         toc = time.clock()
#         elap = toc-tic;
#         print("analyzing image %d / %d. Time: %4.4f seconds." % (i, len(images),elap))
#         tic = time.clock()
#     img, x = load_image(image_path);
#     feat = feat_extractor.predict(x)[0]
#     features.append(feat)
# print('finished extracting features for %d images' % len(images))


# features = np.array(features)
# pca = PCA(n_components=101)
# pca.fit(features)

# pca_features = pca.transform(features)

def get_closest_images(query_image_idx, num_results=5):
    distances = [ distance.cosine(pca_features_t[query_image_idx], feat) for feat in pca_features_t ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

def get_concatenated_images(indexes , thumb_height):
    thumbs = []
    res = []
    for idx in indexes:
        img = image.load_img(images_t[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
        res.append(images_t[idx])
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image, thumbs, res

# load image and extract features


new_image, x = load_image(image_file)
new_features = feat_extractor_t.predict(x)

# project it into pca space
new_pca_features = pca_t.transform(new_features)[0]

# calculate its distance to all the other images pca feature vectors
distances = [ distance.cosine(new_pca_features, feat) for feat in pca_features_t ]
idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:5]  # grab first 5
results_image, pigs, resi = get_concatenated_images(idx_closest, 200)

for i in resi:
    import pandas as pd
    data = pd.read_csv("styles.csv", encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
    ser = i.split('/')[-1].split('.')[0]
    capt = data[data['id'] == int(ser)]
    capt = capt['productDisplayName']
    st.subheader("Results")
    st.image(i, caption=capt, width=300)
    st.write("Pics " + i)
    # st.write(f"Age: {selected_party['age']}")
    # st.write(f"Educational Background: {selected_party['education']}")

import pickle

# pickle.dump(images, open('images.p', 'wb'))
# pickle.dump(pca_features, open('pca_features.p', 'wb'))
# pickle.dump(pca, open('pca.p', 'wb'))
# pickle.dump(feat_extractor, open('feat_extractor.p', 'wb'))


# file1 = open("images.p",'rb')
# images_t = pickle.load(file1)
# file1.close()

# file2 = open("pca_features.p",'rb')
# pca_features_t = pickle.load(file2)
# file2.close()

# file3 = open("pca.p",'rb')
# pca_t = pickle.load(file3)
# file3.close()

# file4 = open("feat_extractor.p",'rb')
# feat_extractor_t = pickle.load(file4)
# file4.close()