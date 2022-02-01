from elasticsearch import Elasticsearch 
import pandas as pd
import numpy as np
from datetime import datetime
from pandas import DataFrame
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)

my_elastic_localhost = 'http://elastic:6bQu0zGe6NjoXOlamVlG@localhost:9200'

def import_csv_to_elastic():
    input_documents = pd.read_csv("movies.csv")
    es=Elasticsearch([my_elastic_localhost])
    es.indices.delete(index='movieid', ignore=[400, 404])
    movieId = input_documents['movieId']
    title = input_documents['title']
    genres = input_documents['genres']

    now = datetime.now()

    for row in range(len(movieId)):
        print(movieId[row])
        print(title[row])
        print(genres[row])
        data = {"movieId": movieId[row], "title": title[row], "genres": genres[row]}
        es.index(index='movieid',doc_type='devops',body=data)

    then = datetime.now()
    print(now)
    print(then)


def search_in_elastic(title):
    es=Elasticsearch([my_elastic_localhost])

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(title)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]  

    filtered_sentence = []  
  
    for w in word_tokens:  
        if w not in stop_words:  
            filtered_sentence.append(w)

    final_input = " ".join(filtered_sentence)
    final_table=[]
    r = es.search(index="movieid", body={"query": {"match": {'title': { 'query': final_input, "operator": "or", 'fuzziness':'auto'}}}}, size=999)
        
    for i in range(len(r['hits']['hits'])):

        sok = r['hits']['hits'][i]
        sok1 = sok['_source']['title'] 
        sok2 = sok['_score']
        sok3 = sok['_source']['movieId']
     
        temp_table=[]
        temp_table.append(sok3)
        temp_table.append(sok2)
        temp_table.append(sok1)
        final_table.append(temp_table)

    df = DataFrame (final_table,columns=['MovieId','Score','Title'])

    return df

def final_task1():
    while True:
        print("\n\nTo terminate the programm type '-1'")
        title = input("Enter your value: ")
        if (title == '-1'):
            break
        df = search_in_elastic(title)
        print(df)

def import_ratings_csv_to_elastic():

    input_documents = pd.read_csv("ratings.csv")

    es=Elasticsearch([my_elastic_localhost])
    es.indices.delete(index='ratings', ignore=[400, 404])

    userId = input_documents['userId']
    movieId = input_documents['movieId']
    rating = input_documents['rating']
    timestamp = input_documents['timestamp']

    now = datetime.now()

    for row in range(len(userId)):
        print(userId[row])
        print(movieId[row])
        print(rating[row])
        print(timestamp[row])
        data = {"userId": userId[row], "movieId": movieId[row], "rating": rating[row], "timestamp": timestamp[row]}
        es.index(index='ratings',doc_type='devops',body=data)

    then = datetime.now()
    print(now)
    print(then)

def calculate_average_movie_rating(movieId):

    es=Elasticsearch([my_elastic_localhost])
    r = es.search(index="ratings", body={"query": {"match": {'movieId':movieId}}}, size=999)
    sum=0
    counter=0
    for i in range(len(r['hits']['hits'])):
        sum = sum + r['hits']['hits'][i]['_source']['rating']
        counter = counter + 1
    if (counter != 0): 
        average = sum/counter
    else:
        average = 0


    return average

def find_user_personal_rating(userId,movieId):

    es=Elasticsearch([my_elastic_localhost])
    r = es.search(index="ratings", body={"query": {"bool": {"must": [{ "match": {'movieId':movieId}},{"match": {'userId':userId}}]}}}, size=999)
    if (r['hits']['total']['value'] == 0):
       rating = 0
    else:
       rating = r['hits']['hits'][0]['_source']['rating']

    return rating

def calculate_combined_score(score,average_rating,user_rating):

    combined_score = 0.5*score + 0.2*average_rating + 0.3*user_rating

    return combined_score


def final_task2():

    while True:

        print("\n\nTo terminate the programm type '-1'")
        title = input("Enter your value: ")

        if (title == '-1'):
            break
        print("\n\nUser id must be a number between 1 to 671")
        user_number = input("Enter your user_id: ")

        if (int(user_number) < 672):

            statistics_table = search_in_elastic(title) 

            statistics_table['Movie_Average_Score'] = ''
            statistics_table['User_Personal_Score'] = ''
            statistics_table['Combined_Score'] = ''

            for i in range(len(statistics_table)):

                temp0 = statistics_table['Score'][i]

                temp1 = calculate_average_movie_rating(statistics_table['MovieId'][i])
                statistics_table.loc[i,'Movie_Average_Score'] = temp1

                temp2 = find_user_personal_rating(user_number, statistics_table['MovieId'][i])
                statistics_table.loc[i,'User_Personal_Score'] = temp2

                temp3 = calculate_combined_score(temp0,temp1,temp2)
                statistics_table.loc[i,'Combined_Score'] = temp3

            sok = statistics_table.sort_values(by=['Combined_Score'], ascending=False)
            sok = sok.reset_index(drop=True)

            print(sok[['MovieId','Score','Movie_Average_Score','User_Personal_Score','Combined_Score']])

        else:
            print('Not Valid User Id')


def ypolozismos_mo_eidous_kathe_user():

    es=Elasticsearch([my_elastic_localhost])
    genre_table=['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','IMAX','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
    list_dict = []
    now = datetime.now()

    sss = 0

    while (sss < 671 ):
        print(sss)
        sss+=1

        r = es.search(index="ratings", body={"query": {"match": {'userId': sss }}}, size=999)

        d = {'Action': {'sum':0, 'counter':0},
            'Adventure': {'sum':0, 'counter':0},
            'Animation': {'sum':0, 'counter':0},
            'Children': {'sum':0, 'counter':0},
            'Comedy': {'sum':0, 'counter':0},
            'Crime': {'sum':0, 'counter':0},
            'Documentary': {'sum':0, 'counter':0},
            'Drama': {'sum':0, 'counter':0},
            'Fantasy': {'sum':0, 'counter':0},
            'Film-Noir': {'sum':0, 'counter':0},
            'Horror': {'sum':0, 'counter':0},
            'IMAX': {'sum':0, 'counter':0},
            'Musical': {'sum':0, 'counter':0},
            'Mystery': {'sum':0, 'counter':0},
            'Romance': {'sum':0, 'counter':0},
            'Sci-Fi': {'sum':0, 'counter':0},
            'Thriller': {'sum':0, 'counter':0},
            'War': {'sum':0, 'counter':0},
            'Western': {'sum':0, 'counter':0},
            'No_Genre': {'sum':0, 'counter':0}
            }

        d_average = {'Action':0 ,
            'Adventure':0,
            'Animation':0,
            'Children':0,
            'Comedy':0,
            'Crime':0,
            'Documentary':0,
            'Drama':0,
            'Fantasy':0,
            'Film-Noir':0,
            'Horror':0,
            'IMAX':0,
            'Musical':0,
            'Mystery':0,
            'Romance':0,
            'Sci-Fi':0,
            'Thriller':0,
            'War':0,
            'Western':0,
            'No_Genre':0
            }

        for i in range(len(r['hits']['hits'])):

            movieId = r['hits']['hits'][i]['_source']['movieId']
            rating = r['hits']['hits'][i]['_source']['rating']

            r1 = es.search(index="movieid", body={"query": {"match": {'movieId': movieId }}}, size=999)
            genre = r1['hits']['hits'][0]['_source']['genres']

            if (genre != '(no genres listed)'):
                bbbb = genre.split("|")

                for j in bbbb:
                    d[j]['sum'] += rating
                    d[j]['counter'] += 1
            else:
                d['No_Genre']['sum'] += rating
                d['No_Genre']['counter'] += 1

    
        for i in genre_table:
            if (d[i]['counter'] != 0):
                d_average[i] = d[i]['sum'] / d[i]['counter']
            else:
                d_average[i] = 0

        list_dict.append(d_average)

        df = pd.DataFrame(list_dict) 
        df.to_csv('output_file.csv', index = False, header = True)


    then = datetime.now()
    print(now)
    print(then)

    return df

def k_means(input_file_csv):

    dataframe_meswn_orwn = pd.read_csv(input_file_csv)

    numpy_meswn_orwn = dataframe_meswn_orwn.to_numpy()

    from sklearn.metrics import silhouette_score
    sil = []
    kmax = 20 
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(numpy_meswn_orwn)
        labels = kmeans.labels_
        ss = silhouette_score(numpy_meswn_orwn, labels, metric = 'euclidean')
        print("\n k=", k, "   ss=", ss)
        sil.append(ss)
    result = np.where(sil == np.amax(sil))
    best_num_of_clusters = int(result[0]) + 2
    print("\nBest number of clusters: ",best_num_of_clusters)
    print("\n\n")

    kmeans = KMeans(n_clusters = best_num_of_clusters).fit(numpy_meswn_orwn)

    return cluster_lables , best_num_of_clusters
def mesos_oros_tainias_meta_clustering_kai_rating_xrhsth(cluster_table , total_num_of_clusters, movieId_input, input_user):

    es=Elasticsearch([my_elastic_localhost])
    r = es.search(index="ratings", body={"query": {"match": {'movieId': movieId_input }}}, size=999)

    final_table = []
    for i in range(len(r['hits']['hits'])):

        temp_table = []

        sok1 = r['hits']['hits'][i]['_source']['userId']
        sok = r['hits']['hits'][i]['_source']['rating']
        sok2 = cluster_table[sok1-1]

        temp_table.append(sok1)
        temp_table.append(sok)
        temp_table.append(sok2)

        final_table.append(temp_table)

    xrhstes_pou_pshfisan_thn_tainia = pd.DataFrame(final_table, columns =['userId', 'rating','clusterId'])

    total_sum = 0

    count_users_that_their_cluster_voted_the_movie = 0

    cluster_table_df = pd.DataFrame(cluster_table, columns = ['clusterId'])

    for i in range(total_num_of_clusters):
        subset_df = xrhstes_pou_pshfisan_thn_tainia[xrhstes_pou_pshfisan_thn_tainia["clusterId"] == i]
        column_sum = subset_df["rating"].sum(axis = 0)
     
        cluster_subset = cluster_table_df[cluster_table_df["clusterId"] == i]
        if (len(subset_df) != 0):
            total_sum += (float(column_sum) / len(subset_df)) * len(cluster_subset)
            count_users_that_their_cluster_voted_the_movie += len(cluster_subset)
            
    movie_average_score_final = float(total_sum) / count_users_that_their_cluster_voted_the_movie

    subset_user_df = xrhstes_pou_pshfisan_thn_tainia[xrhstes_pou_pshfisan_thn_tainia["userId"] == input_user]

    if (len(subset_user_df) == 0 ):
        subset_df = xrhstes_pou_pshfisan_thn_tainia[xrhstes_pou_pshfisan_thn_tainia["clusterId"] == cluster_table[input_user]]
        column_sum = subset_df["rating"].sum(axis = 0)
    
        cluster_subset = cluster_table_df[cluster_table_df["clusterId"] == cluster_table[input_user]]
        if (len(subset_df) != 0):
            user_rating = (float(column_sum) / len(subset_df))
        else:
            user_rating = 0

    else:
        user_rating = subset_user_df["rating"]
      
    return float(movie_average_score_final), float(user_rating)

def final_task3():

    while True:
        print("\n\nTo terminate the programm type '-1'")
        title = input("Enter your value: ")

        if (title == '-1'):
            break
        print("\n\nUser id must be a number between 1 to 671")

        user_number = input("Enter your user_id: ")

        if (int(user_number) < 672):

            statistics_table = search_in_elastic(title) 

            statistics_table['Movie_Average_Score'] = ''
            statistics_table['User_Personal_Score'] = ''
            statistics_table['Combined_Score'] = ''

            cluster_table , total_num_of_clusters = k_means("output_file.csv")


            for i in range(len(statistics_table)):
                tainia, xrhsths = mesos_oros_tainias_meta_clustering_kai_rating_xrhsth(cluster_table , total_num_of_clusters, int(statistics_table['MovieId'][i]), int(user_number))

                temp0 = statistics_table['Score'][i]

                statistics_table.loc[i,('Movie_Average_Score')] = tainia

                statistics_table.loc[i,('User_Personal_Score')] = xrhsths

                temp3 = calculate_combined_score(temp0,tainia,xrhsths)
                statistics_table.loc[i,('Combined_Score')] = temp3


            sok = statistics_table.sort_values(by=['Combined_Score'], ascending=False)
            sok = sok.reset_index(drop=True)

            print(sok[['MovieId','Score','Movie_Average_Score','User_Personal_Score','Combined_Score']])

        else:
            print('Not Valid User Id')
            
def word2vec_titlou():
    from gensim.models import Word2Vec
    import re

    my_list =[]     
    my_id_list = []     
    max_length = 0

    es=Elasticsearch([my_elastic_localhost])
    r = es.search(index="movieid", body={"query": {"match_all": {}}}, size=9999)

    for i in range(len(r['hits']['hits'])):
        my_temp_list = []
        title_without_reg = re.sub('[^A-Za-z0-9-\s]+', "", r['hits']['hits'][i]['_source']['title']) 
        my_temp_list.append(title_without_reg)
        my_list.append(my_temp_list)

        movieid = r['hits']['hits'][i]['_source']['movieId']
        my_id_list.append(movieid)
        
    model = Word2Vec(my_list, min_count=1)

    model_np = np.zeros([1,100])
    for i in my_list:
        sok = model[i]
        model_np = np.append(model_np, sok, axis = 0)
    model_np = np.delete(model_np, 0, 0)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(model_np)
    model_np = scaler.fit_transform(model_np)

    genre_list = []
    for i in range(len(r['hits']['hits'])):
        genres = r['hits']['hits'][i]['_source']['genres']
        if (genres == '(no genres listed)'):
            genres = 'NoGenre'
        genre_list.append(genres)

    final_concatenated_list = []
    for i in range(len(my_list)):
        arr1 = model_np[i]
        arr2 = my_onehot_encoder(genre_list[i])
        final_arr = np.concatenate((arr1,arr2))
        final_concatenated_list.append(final_arr)

    final_concatenated_list = np.array(final_concatenated_list)
    print("\n\nTeliko vector ths 1hs tainias ths listas san paradeigma\n")
    print(final_concatenated_list[0])

    return my_id_list, final_concatenated_list    

def my_onehot_encoder(genre_list_item):

    onehot_dictionary = {'Action':0 ,
            'Adventure':1,
            'Animation':2,
            'Children':3,
            'Comedy':4,
            'Crime':5,
            'Documentary':6,
            'Drama':7,
            'Fantasy':8,
            'Film-Noir':9,
            'Horror':10,
            'IMAX':11,
            'Musical':12,
            'Mystery':13,
            'Romance':14,
            'Sci-Fi':15,
            'Thriller':16,
            'War':17,
            'Western':18,
            'NoGenre':19
            }
    x = genre_list_item.split('|')
    onehot_encoded = np.zeros([20])

    for i in x:
        sok = onehot_dictionary[i]
        onehot_encoded[sok] = 1

    return onehot_encoded

def dhmiourgia_nn_gia_ton_user(userid , movie_id_list, final_concatenated_list):

     es=Elasticsearch([my_elastic_localhost])
     r = es.search(index="ratings", body={"query": {"match": {'userId': userid }}}, size=999)

     movies = []
     movie_ratings = [] 

     for i in range(len(r['hits']['hits'])):
         movies.append(r['hits']['hits'][i]['_source']['movieId'])
         movie_ratings.append(r['hits']['hits'][i]['_source']['rating'])

     user_movie_vectors = np.zeros([1,120])

     movie_id_list = np.array(movie_id_list)

     for i in movies:
        position = np.where(movie_id_list == i)
        user_movie_vectors = np.append(user_movie_vectors, final_concatenated_list[position], axis=0)
     user_movie_vectors = np.delete(user_movie_vectors, 0, 0)

     movie_ratings = np.array(movie_ratings)


     from sklearn.neural_network import MLPClassifier, MLPRegressor
     mlp_1 = MLPRegressor(hidden_layer_sizes=(600,200), alpha=0.0001, random_state=1)
     mlp_1.fit(user_movie_vectors, movie_ratings)

     return mlp_1 

def final_task4():

    while  True:
        print("\n\nTo terminate the programm type '-1'")
        title = input("Enter your value: ")

        if (title == '-1'):
            break
        print("\n\nUser id must be a number between 1 to 671")

        user_number = input("Enter your user_id: ")

        if (int(user_number) < 672):

             statistics_table = search_in_elastic(title) 

             statistics_table['Movie_Average_Score'] = ''
             statistics_table['User_Personal_Score'] = ''
             statistics_table['Combined_Score'] = ''

             movie_id_list, final_concatenated_list = word2vec_titlou()

             mlp = dhmiourgia_nn_gia_ton_user(user_number, movie_id_list, final_concatenated_list)

             cluster_table , total_num_of_clusters = k_means("output_file.csv")

             for i in range(len(statistics_table)):

                 tainia, xrhsths = mesos_oros_tainias_meta_clustering_kai_rating_xrhsth(cluster_table , total_num_of_clusters, int(statistics_table['MovieId'][i]), int(user_number))

                 temp0 = statistics_table['Score'][i]

                 temp1 = tainia                                                             
                 statistics_table.loc[i,'Movie_Average_Score'] = temp1

                 temp2 = find_user_personal_rating(user_number, statistics_table['MovieId'][i])
                 if (temp2 == 0):
                     position = np.where(movie_id_list == statistics_table['MovieId'][i])
                     u = np.empty([1, 120])
                     u[0,:]=final_concatenated_list[position]
                     temp2 = float(mlp.predict(u))
                 statistics_table.loc[i,'User_Personal_Score'] = float(temp2)

                 temp3 = calculate_combined_score(temp0,temp1,temp2)
                 statistics_table.loc[i,'Combined_Score'] = float(temp3)

             sok = statistics_table.sort_values(by=['Combined_Score'], ascending=False)
             sok = sok.reset_index(drop=True)

             print(sok[['MovieId','Score','Movie_Average_Score','User_Personal_Score','Combined_Score']])
        
        else:
            print('Not Valid User Id')


# Main function from where the execution starts
if __name__== "__main__":

 # Initialization
 #import_csv_to_elastic()
 #import_ratings_csv_to_elastic()

 # Initialization for task 3
 #ypolozismos_mo_eidous_kathe_user()

 #final_task1()

 #final_task2()

 #final_task3()

 final_task4()



