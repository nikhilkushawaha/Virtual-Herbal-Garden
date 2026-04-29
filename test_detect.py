import urllib.request
import json
import uuid

def post_multipart(url, filepath):
    import io
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    
    body.write(f'--{boundary}\r\n'.encode('utf-8'))
    body.write(f'Content-Disposition: form-data; name="file"; filename="{filepath}"\r\n'.encode('utf-8'))
    body.write(f'Content-Type: image/jpeg\r\n\r\n'.encode('utf-8'))
    
    with open(filepath, 'rb') as f:
        body.write(f.read())
        
    body.write(f'\r\n--{boundary}--\r\n'.encode('utf-8'))
    
    req = urllib.request.Request(url, data=body.getvalue())
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(e.read().decode())
        return None

res = post_multipart('http://127.0.0.1:8000/detect', 'test_images_rose/rose_00.jpg')
print(json.dumps(res, indent=2))
