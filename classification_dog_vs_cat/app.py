# from flask import Flask, request, render_template_string
# import joblib
# import cv2
# import numpy as np

# app = Flask(__name__)

# # Tải mô hình SVM đã huấn luyện
# model_path = 'best_svm_model.pkl'
# model = joblib.load(model_path)
# image_size = (50, 50)

# def preprocess_image(image):
#     image_resized = cv2.resize(image, image_size)
#     image_normalized = image_resized / 255.0
#     image_flatten = image_normalized.flatten()
#     return np.array([image_flatten])

# upload_html = '''
# <!doctype html>
# <html lang="en">
#   <head>
#     <meta charset="utf-8">
#     <title>Upload Image</title>
#   </head>
#   <body>
#     <h1>Upload an image of a dog or cat</h1>
#     <form method="post" enctype="multipart/form-data">
#       <input type="file" name="file">
#       <input type="submit" value="Upload">
#     </form>
#   </body>
# </html>
# '''

# result_html = '''
# <!doctype html>
# <html lang="en">
#   <head>
#     <meta charset="utf-8">
#     <title>Result</title>
#   </head>
#   <body>
#     <h1>Result: {{ result }}</h1>
#     <a href="/">Upload another image</a>
#   </body>
# </html>
# '''

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             # Đọc và xử lý ảnh
#             img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#             processed_img = preprocess_image(img)
            
#             # Dự đoán
#             prediction = model.predict(processed_img)
#             result = 'Dog' if prediction[0] == 1 else 'Cat'
            
#             return render_template_string(result_html, result=result)
#     return render_template_string(upload_html)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template_string, send_from_directory
import joblib
import cv2
import numpy as np
import os

app = Flask(__name__)

# Tải mô hình SVM đã huấn luyện
model_path = 'best_model.joblib'
model = joblib.load(model_path)
image_size = (50, 50)

def preprocess_image(image):
    image_resized = cv2.resize(image, image_size)
    image_normalized = image_resized / 255.0
    image_flatten = image_normalized.flatten()
    return np.array([image_flatten])

upload_html = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Upload Image</title>
  </head>
  <body>
    <h1>Upload an image of a dog or cat</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" onchange="loadFile(event)">
      <img id="output" width="200">
      <script>
        var loadFile = function(event) {
          var output = document.getElementById('output');
          output.src = URL.createObjectURL(event.target.files[0]);
          output.onload = function() {
            URL.revokeObjectURL(output.src) // free memory
          }
        };
      </script>
      <input type="submit" value="Upload">
    </form>
  </body>
</html>
'''

result_html = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Result</title>
  </head>
  <body>
    <h1>Result: {{ result }}</h1>
    <img src="{{ url_for('uploaded_file', filename=filename) }}" width="200">
    <a href="/">Upload another image</a>
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             # Save the file to a temporary directory
#             filename = 'temp.png'
#             file.save(os.path.join('temp', filename))

#             # Đọc và xử lý ảnh
#             img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#             processed_img = preprocess_image(img)
            
#             # Dự đoán
#             prediction = model.predict(processed_img)
#             result = 'Dog' if prediction[0] == 1 else 'Cat'
            
#             return render_template_string(result_html, result=result, filename=filename)
#     return render_template_string(upload_html)
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the file to a temporary directory
            filename = 'temp.png'
            directory = 'temp'
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = os.path.join(directory, filename)
            file.save(file_path)

            # Đọc và xử lý ảnh
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            processed_img = preprocess_image(img)
            
            # Dự đoán
            prediction = model.predict(processed_img)
            result = 'Dog' if prediction[0] == 1 else 'Cat'
            
            return render_template_string(result_html, result=result, filename=filename)
    return render_template_string(upload_html)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('temp', filename)

if __name__ == '__main__':
    app.run(debug=True)