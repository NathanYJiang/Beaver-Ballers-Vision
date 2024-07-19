import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from facenet_models import FacenetModel

print("hello beaver ballers")

# Create a Profile class with functionality to store face descriptors associated with a named individual.
# Functionality to create, load, and save a database of profiles
# Functionality to add and remove profiles
# Functionality to add an image to the database, given a name (create a new profile if the name isn’t in the database, otherwise add the image’s face descriptor vector to the proper profile)
# manit

#profile stores name and the person's descriptors 
#database stores profiles 
class Profile: 
    #might have to update this based on how we implement. 
    def __init__(self, name, desc) -> None:
        self.name = name #name of person in image
        self.desc = desc #descriptor in array
    
    def __str__(self) -> str:
        return (self.name, len(self.desc))
    
    # def add_desc(self, desc):
    #     self.avg_d = ((self.avg_d * len(self.desc)) + desc) / (len(self.desc) + 1) #recalculates avg based on the new descriptor
    #     self.desc.append(desc) #adds new descriptor to the descriptor array. 

profile_dict = {}

def add_to_dict(profile):
    if profile.name in profile_dict:
        profile_dict[profile.name].append(profile.desc)
    else:
        profile_dict[profile.name] = np.array[profile.desc]

def remove_from_dict_w_profile(profile):
    del profile_dict[profile.name]

def remove_from_dict_w_name(name):
    del profile_dict[name]


# #Database (can't have two things with the same name)
# def create_connection():
#     conn = sqlite3.connect('profiles.db')
#     return conn
# def create_table(conn):
#     with conn:
#         conn.execute('''
#             CREATE TABLE IF NOT EXISTS profiles (
#                 name TEXT NOT NULL,
#                 age INTEGER NOT NULL
#             )
#         ''')

# def add_profile(conn, profile):
#     with conn:
#         conn.execute('conn.execute('INSERT INTO profiles (name, desc) VALUES (?, ?)', (profile.name, profile.desc))')
                     
# def update_profile(conn, name, desc):
#     with conn:
#         conn.execute('UPDATE profiles SET name = ?, desc = ?', (name, desc))
    


# conn = create_connection()
# create_table(conn)




# Function to measure cosine distance between face descriptors. It is useful to be able to take in a shape-(M, D) array of M descriptor vectors and a shape-(N, D) array of N descriptor vectors, and compute a shape-(M, N) array of cosine distances – this holds all MxN combinations of pairwise cosine distances.
# nathan + heidi DONE AND VERIFIED :D
# norm of a vector
def norm(v):
    return np.sqrt(np.sum(v ** 2))

# cosine distance
def distance_metric(d1, d2):
    # M, N size
    distances = np.zeros((d1.shape[0], d2.shape[0]))
    for i, di in enumerate(d1):
        for j, dj in enumerate(d2):
            distances[i, j] = 1 - (di @ dj) / (norm(di) * norm(dj))
    
    return distances


# Estimate a good detection probability threshold for rejecting false detections (e.g. a basketball detected as a face). Try running the face detector on various pictures and see if you notice false-positives (things detected as faces that aren’t faces), and see what the detectors reported “detection probability” is for that false positive vs for true positives.

def get_accuracy(images, labels, prob):
    model = FacenetModel()
    detection_prob = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    idx = 0
    total_pos = 0
    total_neg = 0
    for idx, k in enumerate(detection_prob):
        if k>prob:
            if labels[idx] == 1:
                tp += 1
            else:
                fp += 1
            total_pos += 1
        else:
            if labels[idx] == 0:
                tn += 1
            else:
                fn += 1
            total_neg += 1
    if total_pos != 0:
        true_pos_rate = tp/total_pos
        false_pos_rate = fp/total_pos
    else:
        true_pos_rate = -1
        false_pos_rate = -1
    if false_pos != 0:
        true_neg_rate = tn/total_neg
        false_neg_rate = fn/total_neg
    else:
        true_neg_rate = -1
        false_neg_rate = -1
    overall_accuracy = (tp+tn)/(total_pos+total_neg)
    return overall_accuracy
    
def est_detection_threshold(images, labels, step_size = 0.001):
    prob = 0
    best_acc = 0
    best_acc_prob = 0
    while prob <= 1:
        new_acc = get_accuracy(images, labels, prob)
        if new_acc > best_acc:
            best_acc = new_acc
            best_acc_prob = prob
        prob += step_size
    return best_acc_prob
    
    

# Estimate the maximum cosine-distance threshold between two descriptors, which separates a match from a non-match. Note that this threshold is also needed for the whispers-clustering part of the project, so be sure that this task is not duplicated and that you use the same threshold. You can read more about how you might estimate this threshold on page 3 of this document


# Functionality to see if a new descriptor has a match in your database, given the aforementioned cutoff threshold.


# Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise





