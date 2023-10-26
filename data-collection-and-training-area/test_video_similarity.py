from video_similarity import *

'''
print("Calculating NP Array Similarity in old data: ")
activities = ["gotokal",  "roudro",  "sangbadik"]
X,Y = load_activity_data(activities)
print(cosine_similairty(X[0][0], X[0][1]))
print(cosine_similairty(X[0][0], X[2][1]))
'''


print("Testing Video Similarity between Same Video: ")
print(calculate_video_similarity("demo.mkv","demo.mkv"))