from google.cloud import vision
import io

vision_client = vision.Client()

file_name = '41_5_rgb_286.png'

with io.open(file_name, 'rb') as image_file:
	content = image_file.read()
	image = vision_client.image(content=content)

labels = image.detect_labels()

for label in labels:
	print(label.description)