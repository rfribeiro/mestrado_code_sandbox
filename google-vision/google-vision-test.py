
# coding: utf-8

# In[18]:


from google.cloud import vision
from google.cloud.vision import types
from google.protobuf import text_format
import io


# In[5]:


import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='apikey.json'


# In[7]:


file_name = '41_5_rgb_286.png'


# In[8]:


with io.open(file_name, 'rb') as image_file:
    content = image_file.read()


# In[15]:


client = vision.ImageAnnotatorClient()


# In[19]:


image = types.Image(content=content)


# In[20]:


response = client.face_detection(image=image)


# In[21]:


faces = response.face_annotations


# In[23]:


likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')


# In[24]:


for face in faces:
    print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
    print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
    print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
                for vertex in face.bounding_poly.vertices])

    print('face bounds: {}'.format(','.join(vertices)))


# In[50]:


# Save response to file
f = open('b.txt', 'w')
f.write(text_format.MessageToString(response))
f.close()


# In[49]:


# Read response file
f = open('b.txt', 'r')
resp = types.AnnotateImageResponse() # replace with your own message
text_format.Parse(f.read(), resp)
f.close()


# In[76]:





# In[79]:





# In[82]:




