import os
import time
from flask import Flask, request, flash, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
import cv2

from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
# from speaker_score.speaker_score import VoiceScore
from garbage_predict.customize_service import garbage_predict
from fixation_predict.fixation_predict import fixation_predict
from sod_predict.sod_predict import sod_predict
from matplotlib import pyplot as plt

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads')
PREDICT_FOLDER = os.path.join(os.getcwd(), 'static/predicts')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'bmp', 'png'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICT_FOLDER'] =  PREDICT_FOLDER
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
bootstrap = Bootstrap(app)
# vs = VoiceScore(os.path.join(os.getcwd(), 'speaker_score/ckpt/Speaker_vox_iter_18000.ckpt'))
from spinal_predict.spinal_predict import spinal_predict
from ty_predict.predict import sos_predict
from ty_predict.predict import sod_predict as ty_sod_predict
from PIL import Image
import numpy as np

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('home.html')
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/medical', methods=['GET', 'POST'])
def medical():
    return render_template('medical.html')
@app.route('/sod', methods=['GET', 'POST'])
def sod():
    return render_template('sod.html')
@app.route('/fixation', methods=['GET', 'POST'])
def fixation():
    return render_template('fixation.html')
@app.route('/number', methods=['GET', 'POST'])
def number():
    return render_template('number.html')
@app.route('/competition', methods=['GET', 'POST'])
def competition():
    return render_template('competition.html')


@app.route('/check_spinal', methods=['GET', 'POST'])
def check_spinal():
    if request.method == 'POST':
        img_name = request.files['file']
        secure_img_name = secure_save_file(img_name)
        secure_img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_img_name)
        data_dict = dict(img=secure_img_path)
        out = spinal_predict(data_dict).astype(np.uint8)
        out = Image.fromarray(out)
        out = out.convert('P')
        out.putpalette([0, 0, 0, 255, 0, 0,
                        0, 255, 0, 0, 0, 255])
        out.save(os.path.join(app.config['PREDICT_FOLDER'], secure_img_name))

        return render_template('predict_spinal.html',
                               img_name=secure_img_name,
                               pred_name=secure_img_name)
    return render_template('check_spinal.html')



@app.route('/check_sod', methods=['GET', 'POST'])
def check_sod():
    if request.method == 'POST':
        img_name = request.files['file']
        secure_img_name = secure_save_file(img_name)
        secure_img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_img_name)
        data_dict = dict(img=secure_img_path)
        out = sod_predict(data_dict).astype(np.uint8)
        out = Image.fromarray(out)
        out = out.convert('P')
        out.putpalette([0, 0, 0, 255, 255, 255,
                        0, 255, 0, 0, 0, 255])
        out.save(os.path.join(app.config['PREDICT_FOLDER'], secure_img_name))

        return render_template('predict_sod.html',
                               img_name=secure_img_name,
                               pred_name=secure_img_name)
    return render_template('check_sod.html')


@app.route('/check_garbage', methods=['GET', 'POST'])
def check_garbage():
    if request.method == 'POST':
        img_name = request.files['file']
        secure_img_name = secure_save_file(img_name)
        secure_img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_img_name)
        data_dict = dict(img=secure_img_path)
        out = garbage_predict(data_dict)
        return render_template('predict_garbage.html',
                               img_name=secure_img_name,
                               garbage_cat=out)
    return render_template('check_garbage.html')


@app.route('/check_number', methods=['GET', 'POST'])
def check_number():
    if request.method == 'POST':
        img_name = request.files['file']
        secure_img_name = secure_save_file(img_name)
        secure_img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_img_name)
        data_dict = dict(img=secure_img_path)
        out = sos_predict(data_dict)
        return render_template('predict_number.html',
                               img_name=secure_img_name,
                               sense_number=out)
    return render_template('check_number.html')



@app.route('/check_fixation', methods=['GET', 'POST'])
def check_fixation():
    if request.method == 'POST':
        img_name = request.files['file']
        secure_img_name = secure_save_file(img_name)
        secure_img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_img_name)
        data_dict = dict(img=secure_img_path)
        out = fixation_predict(data_dict)
        plt.axis('off')
        plt.imshow(out, cmap='jet')
        plt.savefig(os.path.join(app.config['PREDICT_FOLDER'], secure_img_name))

        return render_template('predict_fixation.html',
                               img_name=secure_img_name,
                               pred_name=secure_img_name)
    return render_template('check_fixation.html')

class ImageInputForm(FlaskForm):
    img = FileField(u'输入一张图片', validators=[
        DataRequired(u'请输入一个有效的图片')
    ])
    submit = SubmitField(u'检测')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def secure_save_file(file):
    if file and allowed_file(file.filename):
        cur_time = '-'.join(time.ctime().split(' ')).replace(':', '-')
        filename = cur_time + '_' + secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename
    else:
        flash('文件不符合要求')
        return None


if __name__ == '__main__':
    app.run(port=8080, host='0.0.0.0')
