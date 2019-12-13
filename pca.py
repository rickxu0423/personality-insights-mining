import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

idList = ['Openness', 'Adventurousness', 'Artistic Interests', 'Experiencing Emotions', 'Creative Thinking', 'Need for Cognition', 'Questioning', 'Conscientiousness', 'Achievement Striving', 'Cautiousness', 'Dutifulness', 'Orderliness', 'Self-discipline', 'Self-efficacy', 'Extraversion', 'Active', 'Leadership', 'Cheerfulness', 'Need for Stimulation', 'Outgoing', 'Social', 'Agreeableness', 'Altruism', 'Cooperation', 'Modesty', 'Forthright', 'Compassion', 'Trust', 'Emotional Response', 'Easy to Provoke', 'Anxious', 'Despondence', 'Self-control', 'Self-monitoring', 'Stress Management']
labelList = ['Component 1', 'Component 2', 'Component 3']

def pca(file):
    df = pd.read_csv(file)
    x = df.loc[:, idList].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = labelList)
    
    iterables = []
    for label in labelList:
        iterables.append(principalDf[label].values)
    X = np.array(list(zip(*iterables)))
    return X

    