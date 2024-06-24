'''PYTHON MONGODB TUTORIAL USING PYMONGO: https://youtu.be/rE_bJl2GAY8?si=9rwqSn7oI-BkTxsg'''

'''IMPORTS'''
# import pymongo
# from pymongo import MongoClient

'''CONNECTION TO DB AND TABLE'''
# cluster = MongoClient("mongodb+srv://t7bo:t7bo@clusterspeechtotext.kzzprwp.mongodb.net/?retryWrites=true&w=majority&appName=ClusterSpeechToText")
# db = cluster["AzureSpeechToText"]
# collection = db["transcriptions"]

'''CREATION OF A POST/ITEM/ROW IN A TABLE'''
# post1 = {"transcription": "apzazpamamazipdai"}

# post2 = {

#     "audio_file_path": "/djdkkd/djdk",
#     "transcription": "..."
# }

'''INSERT A POST INSIDE DB TABLE'''
# collection.insert_one(post1)

'''INSERT MANY POSTS INSIDE DB TABLE'''
# collection.insert_many([post1, post2])

'''FIND INFO INSIDE TABLE'''
# results = collection.find({"transcription":"..."})
# for result in results:
#     print(result) 

'''FIND ONE INFO (APPARENTLY WILL FIND THE 1ST INFO)'''
# result = collection.find_one({"_id":0})
# print(result)

'''FIND MULTIPLE INFO -> WILL RETURN ALL OBJECTS IN DB TABLE'''
# results = collection.find({})
# for x in results:
#     print(x)

'''UPDATE ITEMS INSIDE TABLE'''
# result = collection.update_one({'_id':7}, {'$set':{"transaction":"abc"}})

'''TOTAL ITEMS INSIDE TABLE'''
# post_count = collection.count_documents({})
# print(post_count)

'''DELETE ONE ITEM INSIDE TABLE'''
# result = collection.delete_one({'_id':0})

'''DELETE EVERYTHING INSIDE TABLE'''
# results = collection.delete_many({})