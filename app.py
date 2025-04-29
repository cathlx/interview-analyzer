import os
from flask import Flask, render_template, request, redirect, send_file
from werkzeug.utils import secure_filename

from processing import analyze

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    try:
        score = analyze(filepath)
        if score > 0: 
            return render_template('internal.html')
        elif score < 0:
            return render_template('external.html')
        else:
            return render_template('mixed.html')
    except Exception as e:
        return render_template('error.html')


@app.route('/download')
def download_example():
    return send_file('data/example.txt', download_name='example.txt', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)