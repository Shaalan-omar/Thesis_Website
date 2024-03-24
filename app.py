from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__,static_url_path='/')


@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/detection')
def detection():
    return app.send_static_file('detection.html')

@app.route('/run-script')
def run_script():
    # Here, you can execute your Python script
    result = "Python script executed successfully!"
    return result

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'Error: No image uploaded', 400

    image_file = request.files['image']
    if image_file.filename == '':
        return 'Error: No selected image', 400

    # Save the uploaded image to a temporary location
    image_path = os.path.join('static', 'uploads', image_file.filename)
    image_file.save(image_path)

    # Execute your Python script with the uploaded image as an argument
    # Replace 'python_script.py' with the name of your Python script
    # Ensure that the Python script is in the same directory as 'app.py'
    os.system(f'python angulation.py {image_path}')

    # Return the filename of the processed image
    processed_image_filename = f'processed_{image_file.filename}'
    return redirect(url_for('display_processed_image', filename=processed_image_filename))

@app.route('/display/<filename>')
def display_processed_image(filename):
    return render_template('display.html', filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
