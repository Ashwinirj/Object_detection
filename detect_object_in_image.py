#importing neccessary packages
import tensorflow as tf
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from PIL import Image, ImageDraw, ImageFont

#path to the input image
image_path='chair.jpg'

#image is read into the image-data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

#The images in the dataset are  represented as graph
label_lines = [line.rstrip() for line in tf.gfile.GFile("tf_files/retrained_labels.txt")]
				   
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
 
    graph_def = tf.GraphDef()	
    graph_def.ParseFromString(f.read())	
    _ = tf.import_graph_def(graph_def, name='')	
	

with tf.Session() as sess:
    #Activation function used for final result 
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \{'DecodeJpeg/contents:0': image_data})
	
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for i,node_id in enumerate(top_k):
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
        
        pred = label_lines[top_k[0]].split(' ')
        img = Image.open(image_path)
        draw  = ImageDraw.Draw(img)
        font  = ImageFont.truetype("abel-regular.ttf", 20,encoding="unic")
        print(type(pred))
        draw.text( (10,150), pred[0], fill='#000000', font=font)
        if i==0:
            img.show()
        
