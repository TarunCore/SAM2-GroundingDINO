# Flask app


# ENDPOINT TO RUN THE CONTAINER
# @params: image will be uploaded
#          prompt will be uploaded
#          
# run the tarune14/sam2-dino container with the parameter for the uploaded image path and prompt
# container will run script.py with the parameters


# ENDPOINT TO GET THE CONTAINER STATUS

from flask import Flask, request, send_file
import os
import docker
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = '/tmp/sam2-dino-uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return {'error': 'No image file provided'}, 400
    
    image = request.files['image']
    prompt = request.form.get('prompt')
    
    if not prompt:
        return {'error': 'No prompt provided'}, 400
    
    if image.filename == '':
        return {'error': 'No selected file'}, 400
    
    # Create temporary files for input and output
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=UPLOAD_FOLDER) as input_file, \
         tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=UPLOAD_FOLDER) as output_file:
        
        # Save uploaded image
        image.save(input_file.name)
        
        # Initialize Docker client
        client = docker.from_env()
        
        try:
            # Run the container
            container = client.containers.run(
                'tarune14/sam2-dino:latest',
                command=[
                    'python', 'script.py',
                    '--image_path', '/app/input.png',
                    '--prompt', prompt,
                    '--output_path', '/app/output.png'
                ],
                volumes={
                    input_file.name: {'bind': '/app/input.png', 'mode': 'ro'},
                    output_file.name: {'bind': '/app/output.png', 'mode': 'rw'}
                },
                detach=True
            )
            
            # Wait for container to finish
            result = container.wait()
            
            if result['StatusCode'] != 0:
                return {'error': 'Processing failed'}, 500
            
            # Return the output image
            return send_file(output_file.name, mimetype='image/png')
            
        except Exception as e:
            return {'error': str(e)}, 500
        
        finally:
            # Cleanup
            try:
                container.remove()
            except:
                pass
            os.unlink(input_file.name)
            os.unlink(output_file.name)

@app.route('/health', methods=['GET'])
def health_check():
    return {'status': 'healthy'}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

