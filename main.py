import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from facenet_models import FacenetModel

print("welcome beaver ballers")

# class Profile: 
#     #might have to update this based on how we implement. 
#     def __init__(self, name, desc) -> None:
#         self.name = name #name of person in image
#         self.desc = np.array([desc]) #descriptor in array
    
#     def __str__(self) -> str:
#         return (self.name, len(self.desc))


#     def add_to_dict(self, desc):
#         if self.name in profile_dict:
#             profile_dict[self.name].append(self.desc)
#         else:
#             profile_dict[self.name] = np.array([self.desc])

#     def remove_from_dict_w_profile(self):
#         del profile_dict[self.name]


    # def remove_from_dict_w_name(name):
    #     del profile_dict[name]

    # def add_desc(self, desc):
    #     self.avg_d = ((self.avg_d * len(self.desc)) + desc) / (len(self.desc) + 1) #recalculates avg based on the new descriptor
    #     self.desc.append(desc) #adds new descriptor to the descriptor array. 









# Estimate a good detection probability threshold for rejecting false detections (e.g. a basketball detected as a face). Try running the face detector on various pictures and see if you notice false-positives (things detected as faces that aren’t faces), and see what the detectors reported “detection probability” is for that false positive vs for true positives.
from PIL import Image, ImageDraw, ImageFont

model = FacenetModel()

def DrawBoxOnPicture(image, box, name, text, output_filename = "output_image.jpg"):
    draw = ImageDraw.Draw(image)
    font_size = int((box[2] - box[0]) / 6)
    font = ImageFont.truetype("Arial Unicode.ttf", font_size)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
    text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
    draw.rectangle([text_bbox[:2], text_bbox[2:]], fill="red")
    draw.text((box[0], box[1]), text, fill="white", font=font)

    # name line
    draw.text((box[0], box[3]), name, fill="white", font=font)

    image.show()
    #image.save(output_filename)

def DrawBoxesOnPicture(image, boxes, labels):
    for box, text in zip(boxes, labels):
        DrawBoxOnPicture(image, box, text)

def process_image(i_p):
    image_path = i_p
    image = Image.open(image_path)

    #show the image first
    image.show()
    image = image.convert('RGB')
    image_array = np.array(image)
    boxes, probabilities, landmarks = model.detect(image_array) 
    #boxes: top left, bottom right (x,y and starts from 0,0 as top left of full image)
    boxes = boxes[probabilities > 0.95]
    #fltering out non humans

    descriptors = model.compute_descriptors(image_array, boxes)
    # descs shape (1, 512)

    name = check_match(descriptors)    
    
    #print(image_array.shape)
    #print("boxes: ", boxes, "probs: ", probabilities)

    # For both step 4 and 7
    if boxes is None:
        print("No Faces Detected")
    else:
        # For step 4
        for box, prob in zip(boxes, probabilities):
            DrawBoxOnPicture(image, box, name, f"{prob:.2f}" + "")

        # Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise
        #use M

# Create a Profile class with functionality to store face descriptors associated with a named individual.
# Functionality to create, load, and save a database of profiles
# Functionality to add and remove profiles
# Functionality to add an image to the database, given a name (create a new profile if the name isn’t in the database, otherwise add the image’s face descriptor vector to the proper profile)
# manit

#profile stores name and the person's descriptors 
#database stores profiles 


profile_dict = {}


class Profile:
    def __init__(self, name, desc) -> None:
        self.name = name #name of person in image
        self.desc = np.array([desc]) #descriptor in array
        profile_dict[self.name] = self
    def __repr__(self):
        return str("Name: ", self.name, "\nDescs:", len(self.desc))

    def add_desc(self, desc):
        #self.desc shape (1, 1, 512)
        # we want to append (1, 512)
        self.desc = np.append(self.desc, np.array([desc]), axis=0)
        
    def remove_from_dict(self):
        del profile_dict[self.name]

# Function to measure cosine distance between face descriptors. It is useful to be able to take in a shape-(M, D) array of M descriptor vectors and a shape-(N, D) array of N descriptor vectors, and compute a shape-(M, N) array of cosine distances – this holds all MxN combinations of pairwise cosine distances.
# nathan + heidi DONE AND VERIFIED :D
# norm of a vector
def norm(v):
    return np.sqrt(np.sum(v ** 2))

# cosine distance
def distance_metric(d1, d2):
    # M, N size
    distances = np.empty((d1.shape[0], d2.shape[0]))
    for i, di in enumerate(d1):
        for j, dj in enumerate(d2):
            distances[i, j] = 1 - (di @ dj) / (norm(di) * norm(dj))
    
    return distances
    

# Estimate the maximum cosine-distance threshold between two descriptors, which separates a match from a non-match. Note that this threshold is also needed for the whispers-clustering part of the project, so be sure that this task is not duplicated and that you use the same threshold. You can read more about how you might estimate this threshold on page 3 of this document

cutoff = 0.5

# Functionality to see if a new descriptor has a match in your database, given the aforementioned cutoff threshold.
#iterate through the values
#find average of all current desccriptors
# compare with loaded descriptor
# if below cutoff then add
# if above cutoff then tell user its unknown and prompt them to ask for a name, default to unknown
def check_match(desc):
    for key, value in profile_dict.items():
        avg_value = np.mean(value.desc, axis=0)
        # the number one debug line of all time: print("wahoo", key, value.desc.shape, desc.shape)
        if (distance_metric(avg_value, desc) < cutoff):
            value.add_desc(desc)
            return key
        
    name = input('We do not recognize this face. Enter a name or press enter for UNKNOWN: ')
    if not name:
        name = 'UNKNOWN'
    
    Profile(name, desc)
    return name
    


# process_image('./celebrity_photos/Carlsen/Carlsen-1.jpeg')

# process_image('./celebrity_photos/Drake/drake-3.jpg')
# # process_image('./celebrity_photos/Kendrick/kendrick-1.jpg')
# # process_image('./celebrity_photos/Sabrina/sabrina-1.jpg')

# process_image('./celebrity_photos/Carlsen/Carlsen-2.jpeg')
# process_image('./celebrity_photos/Drake/drake-2.jpg')
# process_image('./celebrity_photos/Carlsen/Carlsen-3.jpeg')
# process_image('./celebrity_photos/Drake/drake-5.jpg')
# process_image('./celebrity_photos/Carlsen/Carlsen-4.jpeg')
# process_image('./celebrity_photos/Drake/drake-4.jpg')

for i in range(4):
    process_image(f'./celebrity_photos/nathan/Nathan-{i + 1}.jpg')
    process_image(f'./celebrity_photos/Drake/drake-{i + 1}.jpg')



print('end')