import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory,flash 
from werkzeug.utils import secure_filename
import cv2
import numpy as np

import numpy as np
import tensorflow as tf 
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.nn import Sequential, Module
from torchvision import transforms
from torchvision.utils import save_image
from model import Model
from PIL import Image, ImageStat
from  style_transfer import *


app = Flask(__name__)
UPLOAD_FOLDER = './static/image/upload/'
filename = 'target.png'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


style =""
@app.route("/")
def home():
	return render_template("index.html")


@app.route("/success", methods=['POST'])

def upload_file():
			content = request.files['file']
			style = request.form.get('style')
			content.save(os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg'))
			#load in content and style image
			content = './static/image/upload/content.jpg'
		 	#Resize style to match content, makes code easier
			stylep = './static/image/s'+ style+'.jpg'
			#Generate image
			output = adain(content, stylep)
			#applysimoserra
			sim_path = 'static/image/upload/content_s'+style+'.png'
			sim = simplify_simo(sim_path,filename)
			data={
                "processed_img":'static/image/upload/content_s'+style+'.png',
                "uploaded_img": './static/image/upload/content.jpg',
				"sim_img":'static/image/upload/target.png'
            }

			return render_template('success.html', data=data)


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def adain(content_path,style_path,alpha = 1):
    if torch.cuda.is_available() :
        device = torch.device(f'cuda:0')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model()
    
    model.load_state_dict(torch.load("./saved_models/model_state.pth", map_location=lambda storage, loc: storage))
    model = model.to(device)

    c = Image.open(content_path)
    s = Image.open(style_path)
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor, alpha)
    
    out = denorm(out, device)

    
    c_name = os.path.splitext(os.path.basename(content_path))[0]
    s_name = os.path.splitext(os.path.basename(style_path))[0]
    output_name = f'{c_name}_{s_name}'

    save_image(out, f"{UPLOAD_FOLDER}{output_name}.png", nrow=1)
    o = Image.open(f"{UPLOAD_FOLDER}{output_name}.png")
    return o   



def simplify_simo(path,filename):
    model_import = __import__("model_gan", fromlist=['model', 'immean', 'imstd'])
    model = model_import.model
    immean = model_import.immean
    imstd = model_import.imstd

    use_cuda = torch.cuda.device_count() > 0

    model.load_state_dict(torch.load("./saved_models/model_gan" + ".pth"))
    model.eval()


    data = Image.open(path).convert('L')
    w, h = data.size[0], data.size[1]
    pw = 8 - (w % 8) if w % 8 != 0 else 0
    ph = 8 - (h % 8) if h % 8 != 0 else 0
    stat = ImageStat.Stat(data)

    data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
    if pw != 0 or ph != 0:
        data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

    if use_cuda:
        pred = model.cuda().forward(data.cuda()).float()
    else:
        pred = model.forward(data)
    save_image(pred[0], f"{UPLOAD_FOLDER}{filename}")
    sim = cv2.imread(f"{UPLOAD_FOLDER}{filename}")
    sim = cv2.cvtColor(sim,cv2.COLOR_BGR2RGB)
    return sim
							

if __name__ =="__main__":
	app.run(debug=True)
