from video_similarity import *

activities = ["gotokal",  "roudro",  "sangbadik"]
X,Y = load_activity_data(activities)
print(cosine_similairt(X[0][0], X[0][1]))
print(cosine_similairt(X[0][0], X[2][1]))