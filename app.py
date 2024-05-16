from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
import json

app = Flask(__name__,static_url_path='/')


@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/detection')
def detection():
    return app.send_static_file('detection.html')

@app.route('/about')
def about():
    return app.send_static_file('about.html')

@app.route('/contact')
def contact():
    return app.send_static_file('contact.html')

@app.route('/team')
def team():
    return app.send_static_file('team.html')

@app.route('/run-script')
def run_script():
    # Here, you can execute your Python script
    result = "Python script executed successfully!"
    return result

@app.route('/detect-image', methods=['POST'])
def detect_impact():
    filename = request.form.get('filename')

    image_path = os.path.join('static', 'processed', filename)

    os.system(f'python Detect.py {image_path}')
    
    json_output_path = os.path.join('static', 'processed', filename.replace('.jpg', '.json'))
    with open(json_output_path, 'r') as f:
        results = json.load(f)

    # return redirect(url_for('display_processed_image', filename=processed_image_filename, results=results))
    return render_template('Result.html', filename=filename, results=results)


@app.route('/AI-detected', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'Error: No image uploaded', 400

    image_file = request.files['image']
    if image_file.filename == '':
        return 'Error: No selected image', 400

    # Save the uploaded image to a temporary location
    image_path = os.path.join('static', 'uploads', image_file.filename)
    image_file.save(image_path)

    os.system(f'python YOLO.py {image_path}')

    # Return the filename of the processed image
    processed_image_filename = f'{image_file.filename}'
    
    return redirect(url_for('display_processed_image', filename=processed_image_filename))


@app.route('/display/<filename>')
def display_processed_image(filename):
    return render_template('display.html', filename=filename)


if __name__ == '__main__':
    app.run(debug=True)