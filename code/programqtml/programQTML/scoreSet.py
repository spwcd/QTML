import numpy as np

def bordaScore(lNum):
    #lNum = [a, b, c, d, e]                                      # creat a list on numbers
    rank = sorted(range(len(lNum)), key=lambda k: lNum[k])      # find order index of the list     
    rank = np.array(rank)
    sizeNum = len(lNum)
    score = np.zeros(sizeNum)
    diffRank = np.zeros(sizeNum-1)
    point = 0; 
    
    # assign score 
    for i in range(0,sizeNum):
        score[rank[i]] = point
        point = point+1
        
    
    # dealing with same values
    for i in range(0,sizeNum-1):
        diffRank[i] = lNum[rank[i+1]] - lNum[rank[i]] 
    
    i=0
    total = 0
    count = 0
    while i<sizeNum-1:                                        # loop all dffRank
        if diffRank[i] == 0:                                    # find the first one with diffRank 0
            start = i
            j = i
            while j<sizeNum-1:                                  # loop to find consequeced 0
                total = total + score[rank[j]]
                count = count + 1
                j = j + 1
                if j<sizeNum-1:
                    if diffRank[j] != 0:                            # the end of consequenced 0
                        stop = j
                        total = total + score[rank[j]]
                        count = count + 1
                        newScore = total/count
                        for k in range(start,stop):
                            score[rank[k]] = newScore
                            score[rank[k+1]] =  newScore
                        i=j
                        total = 0
                        newScore = 0
                        count = 0
                        break
                else:
                    stop = j
                    total = total + score[rank[j]]
                    count = count + 1
                    newScore = total/count
                    for k in range(start,stop):
                            score[rank[k]] = newScore
                            score[rank[k+1]] =  newScore
                        
                
        i = i+1
        
    return score
               