# -*- coding: utf-8 -*-
"""
This is a program that helps provide users with movie recommendations based on the
ratings of similar users. The dataset used for this program is the MovieLens dataset
which is a dataset containing around 100k ratings of users. If we want to predict
the movie recommendations for a user , oone thing that can be done is to look at the 
recommendations given by similar users. This is precisely the idea behind collaborative 
fitering . The neighbourhood model is used calculate top N neighbours and based on the similarity
as a weight assigned to the neighbours the movie rating of a given user can be calculated.


"""

import numpy as np
import scipy
import pandas as pd
from scipy.spatial.distance import correlation

class MovieRecommendations : 
    
    def __init__(self, movieInfoFile,dataFile):
        self.data = pd.read_csv(dataFile,sep='\t',header=None,names=['userId','itemId','rating','timestamp'])
        self.movieInfoFile=movieInfoFile
        movieInfoData = pd.read_csv(self.movieInfoFile,sep='|',header=None,index_col=False,names=['itemId','title'],usecols=[0,1])
        self.movieInfoData=movieInfoData
        
        
        
    def creatUserRatingMatrix(self) :
        self.data = pd.merge(self.data,self.movieInfoData,left_on='itemId',right_on='itemId')
#        self.data.head()
        self.userRatingsMatrix = pd.pivot_table(self.data,values='rating',index=['userId'],columns=['itemId'])
       # self.userRatingsMatrix.head()
       
    def similarity(self,user1,user2) : 
        #normalize by the mean rating of user 1
        #this removes any biases of that user
        #nan mean will ignore any 
        user1 = np.array(user1) - np.nanmean(user1)
        user2 = np.array(user2) - np.nanmean(user2)
        
        #now we have two vectors that are adjusted by the mean ratings of the users
        #now lets look at the similarity
        #now we have to subset each user to give us only those ratings for each user for which both the users have provided a rating
        #commonItemIds gives us the list of item id's or movies for which we have ratings from both user 1 and 2
        #commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
        commonItemIds = [i for i in range(len(user1)) if user1[i] > 0 and user2[i] > 0 ]
        
        if len(commonItemIds) == 0 :
            #no movie in common
            return 0
        else :
            #subset both the users to contain only the elments common to both
            user1 = np.array([user1[i] for i in commonItemIds] )
            user2 = np.array([user2[i] for i in commonItemIds] )
            return correlation(user1,user2)
    
            
            
    #using this similarity definition we can compute the similarity between the active user and every other user and then find the 10 nearest neighbours
    # this function will compute the K nearest neighbours of the current user 
    # then we can use the ratings given by thise k nearest neighbours to predict the ratings given by the current user
    
    def nearestNeighbourRating(self,currentUser,K) : 
        #similarityMatrix is a matrix whose rows represent all the users and values represent similarity of that user
        #with the current user
        similarityMatrix = pd.DataFrame(index=self.userRatingsMatrix.index,columns=['Similarity'])
        
        for i in self.userRatingsMatrix.index : 
            similarityMatrix.loc[i] = self.similarity(self.userRatingsMatrix.loc[currentUser],
                                                  self.userRatingsMatrix.loc[i]  )
        #similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,
                                                 # ['Similarity'],ascending=[0])  
        similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
        #similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=False) 
        nearestNeighbours=  similarityMatrix[:K]
        #We will now use the ratings of the K nearest neighbours found above to predict the current user's ratings
        neighborRatings=self.userRatingsMatrix.loc[nearestNeighbours.index]
        
        predictItemRating= pd.DataFrame(index = self.userRatingsMatrix.columns , columns=['Rating'] )
        #A place holder for the predicted Item ratings. Its row index is the list of item ids which is same 
        #as the column index of userRating matrix
        for i in self.userRatingsMatrix.columns:
            #predictedRating's default value is the average rating of the current user and in case the movie is not seen 
            #by any of the neighbouring users this is the value of the predictedRating for that movie
            predictedRating = np.nanmean(self.userRatingsMatrix.loc[currentUser])
            for j in neighborRatings.index:
                # consider each neighbouring user in the list
                #if the neighbouring user's rating for ith movie is > 0 then :
                if self.userRatingsMatrix.loc[j,i] > 0:
                    #if this is the case then add the neighbor(j)'s rating of the ith item to this predicted rating
                    #also adjust this value with the mean rating of that neighbour
                    #and that difference is weighted by the similarity of this jth user to the current user
                    predictedRating+= (self.userRatingsMatrix.loc[j,i]-np.nanmean(self.userRatingsMatrix.loc[j])) * nearestNeighbours.loc[j,'Similarity']
            #the predictedRating mateix will have the predicted rating for every item for the active/current user
            predictItemRating.loc[i,'Rating'] = predictedRating
        return  predictItemRating
                
            
            
    def topNRecommendations(self,currentUser,N) : 
       # print "hellp"
        #we'll remove the list of movies that the active user has already watched
        predictItemRating = self.nearestNeighbourRating(currentUser,10)
        moviesWatched  =  list(self.userRatingsMatrix.loc[currentUser].loc[self.userRatingsMatrix.loc[currentUser]>0].index)
        #find the list of movies for which the user has already given the rating
        predictItemRating = predictItemRating.drop(moviesWatched)
        topRecommendations = pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:N]
        #This will return a list of ids
        
        topReccTitles=(self.movieInfoData.loc[self.movieInfoData.itemId.isin(topRecommendations.index)])
       # print(topReccTitles)
        return list(topReccTitles.title)
            
        
        #lets print top n reccomendations of a user
    def favoriteMovies(self,activeUser,N):
        #1. subset the dataframe to have the rows corresponding to the active user
        # 2. sort by the rating in descending order
        # 3. pick the top N rows
        topMovies=pd.DataFrame.sort_values(
            self.data[self.data.userId==activeUser],['rating'],ascending=[0])[:N]
        # return the title corresponding to the movies in topMovies 
        return list(topMovies.title)
            
            
            
    def recommendMovies(self,activeUser):
        print self.favoriteMovies(activeUser,5) ,"\n",self.topNRecommendations(activeUser,3)
        
            
    
        
    def displayData(self):
        print 'movie data' 
        print self.movieInfoData.head()
        print ' user data'
        print self.data.head()
        print 'user ratings data'
        print self.userRatingsMatrix.head()
    
    

mov1 = MovieRecommendations('./ml-100k/u.item','./ml-100k/u.data')
#merge the movies database and the user ratings data to create a matrix of user rows and the movies as columns and each cell represents a rating
mov1.creatUserRatingMatrix()
#find the top recommendations for user with id 5
mov1.recommendMovies(5)
