import json, csv

idList = ['Openness', 'Adventurousness', 'Artistic Interests', 'Experiencing Emotions', 'Creative Thinking', 'Need for Cognition', 'Questioning', 'Conscientiousness', 'Achievement Striving', 'Cautiousness', 'Dutifulness', 'Orderliness', 'Self-discipline', 'Self-efficacy', 'Extraversion', 'Active', 'Leadership', 'Cheerfulness', 'Need for Stimulation', 'Outgoing', 'Social', 'Agreeableness', 'Altruism', 'Cooperation', 'Modesty', 'Forthright', 'Compassion', 'Trust', 'Emotional Response', 'Easy to Provoke', 'Anxious', 'Despondence', 'Self-control', 'Self-monitoring', 'Stress Management']

def transfer(personality):
    dict_personality = json.loads(personality)
    #idArray = []
    percentageArray = []
    for i in range(5):
        #idArray.append(dict_personality['tree']['children'][0]['children'][0]['children'][i]['id'])
        percentageArray.append(float(dict_personality['tree']['children'][0]['children'][0]['children'][i]['percentage']))
        for r in range(6):
            #idArray.append(dict_personality['tree']['children'][0]['children'][0]['children'][i]['children'][r]['id'])
            percentageArray.append(float(dict_personality['tree']['children'][0]['children'][0]['children'][i]['children'][r]['percentage']))
    return percentageArray


while 1:
    dataFile = "data/personality_raw.csv"   #input("Enter file name: ")
    if dataFile:
        try:
            file = open(dataFile)
            break
        except:
            print("File not exists!")

with open("data/personality.csv", "w") as csvfile:
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(idList)
    for line in file:
        if not line:
            continue
        personality = line.strip()
        percentageArray = transfer(personality)
        filewriter.writerow(percentageArray)
