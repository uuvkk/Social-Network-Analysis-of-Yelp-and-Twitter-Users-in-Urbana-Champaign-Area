import json, csv
import numpy as np
from sys import getsizeof
from collections import Counter

idx = dict()
ivt_idx = dict()

def get_idx():
    with open("try/urbana_champaign/user_id.json", 'r') as f:
        user_ids = json.loads(f.readline())

    cur = 0
    for id in user_ids:
        idx[id] = cur
        ivt_idx[cur] = id
        cur += 1
        
def to_csv():
    adj = np.full((cur,cur), False, dtype=bool)
    cnt = 0
    with open("../user.json", 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['user_id'] not in idx:
                continue
            user = idx[data['user_id']]

            for friend in data['friends']:
                if friend in idx:
                    adj[user, idx[friend]] = True
    
    w = csv.writer(open("index.csv", "w"))
    for key, val in ivt_idx.items():
        w.writerow([key, val])
    np.savetxt("friends.csv", adj, fmt="%i", delimiter=",")


def to_json():
    outfile = open('adj_list.txt', 'w')
    cnt = 0
    with open("../user.json", 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['user_id'] not in idx:
                continue

            newdata = dict()
            newdata['user_id'] = data['user_id']
            newdata['friends'] = []
            for friend in data['friends']:
                if friend in idx:
                    newdata['friends'].append(friend)
            outfile.write(json.dumps(newdata))
            outfile.write('\n')

            # cnt += 1
            # if cnt== 100:
            #     break


# get_idx()
# to_json()
# exit()

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def friendship():
    ## create the friendship network
    # read in adjacency list
    G = nx.Graph()
    with open("output/adj_friendship.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            u = data['user_id']
            friend = data['friends']
            for v in friend:
                G.add_edge(u,v)
                
    # draw nodes, edges and labels
    df_node_degree = pd.DataFrame(list(dict(G.degree()).items()), columns=["node_name", "degree"])
    # print(df_node_degree.sort_values("degree", ascending=False).head(10))

    # select the biggest component for the following analysis
    connected_components = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    print("{} connected components found.".format(len(connected_components)))
    g = connected_components[0]
    print(g.number_of_nodes())

    # degree distribution plot of the biggest component
    fig1, ax1 = plt.subplots(1,1)
    degree_values = [v for k,v in g.degree()]
    ax1.hist(list(degree_values), bins=list(range(max(degree_values))), log=True)
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Degree Distribution")

    fig1.savefig('/home/dtao2/Dropbox/degree.png')

    # friendship network plot of the biggest component
    fig2, ax2 = plt.subplots(1,1)
    nx.draw_networkx(
        connected_components[0], with_labels=False,
        node_size=[x[1]*10 for x in connected_components[0].degree()],
        pos=nx.spring_layout(connected_components[0]),
        width=0.5,
        ax=ax2
    )
    ax2.axis("off")
    ax2.set_title("Friendship Network")
    fig2.savefig('/home/dtao2/Dropbox/graph.png')

    # # lastly, save file the user_id in the component graph
    
    # outfile = open("output/component_adjlist.txt", 'w')
    # ids = set()
    # for id in g.nodes():
    #     ids.add(id)
    # with open("output/adj_friendship.txt", 'r') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         user_id = data['user_id']
    #         if user_id in ids:
    #             outfile.write(line)
                
    exit()

    # d = g.degree()
    # for i, k in enumerate(d):
    #     print(k)
    #     if i > 10:
    #         break

    # print("radius: {:d}\n".format(nx.radius(g)))
    # print("diameter: {:d}\n".format(nx.diameter(g)))
    # print("eccentricity: {}\n".format(nx.eccentricity(g)))
    # print("center: {}\n".format(nx.center(g)))
    # print("periphery: {}\n".format(nx.periphery(g)))
    # print("density: {:f}".format(nx.density(g)))
                
    return

def user_user():
    # create the user list for searching, the number of user_id is 2454
    user_ids = []
    user_friends = []
    with open("output/component_adjlist.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            user_ids.append(data['user_id'])
            user_friends.append(data)
    
    # create the user_biz list for the biggest component
    user_biz = []
    with open("output/user_biz.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['user_id'] in user_ids:
                user_biz.append(data)

    # print(len(user_biz))
    # print(user_friends[10])
    # print(user_ids[10])
    # print(user_biz[10])
    
    n = len(user_ids)
    matrix = np.zeros((n,n))
    matrix_friends = []
    matrix_unfriends = []
    iter_1 = 0
    iter_2 = 0
    sum_1 = 0
    sum_2 = 0
                
    # for i in range(n):
    #     for j in range(n):
    #         if i != j:
                       
    #             res_i = set(user_biz[i]['biz_id'])
    #             res_j = set(user_biz[j]['biz_id'])
    #             # get the number of same restaurants two users reviewd
    #             num_res_sharing = len(res_i.intersection(res_j))
    #             # set a threshold 
    #             if num_res_sharing > 0:
    #                 matrix[i][j] = num_res_sharing
                    
    # np.savetxt("output/covisiting_matrix", matrix, fmt='%d')

    for i in range(n):
        for j in range(n):
            if i != j:
                
                # compute the number of co-reviewed restaurants for each pair of friends
                if user_biz[j]['user_id'] in user_friends[i]['friends']:
                    res_i = set(user_biz[i]['biz_id'])
                    res_j = set(user_biz[j]['biz_id'])
                    # get the number of same restaurants two users reviewd
                    num_res_sharing = len(res_i.intersection(res_j))
        
                    # matrix_friends.append(num_res_sharing)
                    # iter_1 += 1
                    # sum_1 += num_res_sharing
                    
                    num_total = len(res_i.union(res_j))
                    if num_total > 0:
                        proportion = round(num_res_sharing/num_total, 2)
                        matrix_friends.append(proportion)
                        iter_1 += 1
                        sum_1 += proportion
                    
                else:
                    res_i = set(user_biz[i]['biz_id'])
                    res_j = set(user_biz[j]['biz_id'])
                    # get the number of same restaurants two users reviewd
                    num_res_sharing = len(res_i.intersection(res_j))
                    
                    # matrix_unfriends.append(num_res_sharing)
                    # iter_2 += 1
                    # sum_2 += num_res_sharing
                    
                    num_total = len(res_i.union(res_j))
                    if num_total > 0:
                        proportion = round(num_res_sharing/num_total, 2)
                        matrix_unfriends.append(proportion)
                        iter_2 += 1
                        sum_2 += proportion
                    
    aver_1 = sum_1/iter_1
    aver_2 = sum_2/iter_2
    
    print("similarity of reviewing restaurants for people who are friends: ", aver_1)
    print("similarity of reviewing restaurants for people who are not friends: ", aver_2)
    
    exit()
    
    # degree distribution plot of the biggest component
    fig1, ax1 = plt.subplots(1,1)
    degree_values = matrix_unfriends
    ax1.hist(list(degree_values), bins=list(range(max(degree_values))), log=True)
    
    ax1.set_xlim([0,35])
    ax1.set_ylim([1,10000000])
    
    ax1.set_xlabel("Similarity of coreviewed restaurants")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Non-friend coreviewing distribution")
    
    # ax1.text(50, 300, 'Average = '+str(round(aver_1,2)), fontsize=12)
    
    fig1.savefig('/home/dtao2/Dropbox/coreview_unfriends.png')
    return 


def small_graph():
    #read in the raw matrix
    A = np.loadtxt("output/covisiting_matrix")
    id = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if int(A[i][j]) > 0:
                id.add(i)
                id.add(j)
    id_sorted = sorted(list(id))

    # find the restaurant_ids for each node in this network
    user_ids = []
    with open("output/component_adjlist.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            user_ids.append(data['user_id'])
    
    # create the user_biz list for the biggest component
    user_biz = []
    with open("output/user_biz.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['user_id'] in user_ids:
                user_biz.append(data)
    type = {}
    type['user'] = []

    # creat 3 buckets to separate users by their food_type
    estern = []
    western = []
    middle = []

    # count the categories of restaurants for each user
    matrix = np.ndarray((len(id_sorted), 2), dtype = float)
    for i in range(len(id_sorted)):
   
        id = id_sorted[i]
        biz = user_biz[id]['biz_id']
        user = user_biz[id]['user_id']
        categories = []
        for b in biz:
            with open("output/biz_category.txt", 'r') as filename:
                for line in filename:
                    data = json.loads(line)
                    if data['biz'] == b:
                        for j in data['categories']:
                            categories.append(j)
        cnt = Counter()
        sum = 0
        for word in categories:
            cnt[word] += 1
            sum += 1
        western_res = cnt['American (Traditional)']+cnt['American (New)']+cnt['Italian']+cnt['Mexican']
        estern_res = cnt['Thai']+cnt['Chinese']+cnt['Japanese']+cnt['Korean']+cnt['Indian']
        sum = western_res + estern_res
        if sum != 0:
            pro_western = round(western_res/sum, 2)
            pro_estern = round(estern_res/sum, 2)
            food_type = [pro_western, pro_estern]
            matrix[i] = [pro_western, pro_estern]
            print(pro_western, pro_estern)
      
    # np.savetxt("output/xy.csv", matrix, fmt='%f', delimiter = ',')
    
    
            # identify 3 regions of users
            if pro_western < 0.2:
                estern.append(user)
            elif pro_western > 0.8:
                western.append(user)
            else:
                middle.append(user)


    # write 3 files estern/western/middle to create 3 friendship networks
    outfile_estern = open('output/estern.txt', 'w')
    outfile_western = open('output/western.txt', 'w')
    outfile_middle = open('output/middle.txt', 'w')
    
    with open("output/component_adjlist.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            user_id = data['user_id']
            friends = data['friends']
            if user_id in estern:
                estern_friend = dict()
                estern_friend['user_id'] = user_id
                estern_friend['friends'] = []
                for friend in friends:
                    if friend in estern:
                        estern_friend['friends'].append(friend)
                outfile_estern.write(json.dumps(estern_friend)+'\n')

                        
            elif user_id in western:
                western_friend = dict()
                western_friend['user_id'] = user_id
                western_friend['friends'] = []
                for friend in friends:
                    if friend in western:
                        western_friend['friends'].append(friend)
                outfile_western.write(json.dumps(western_friend)+'\n')

            else:
                middle_friend = dict()
                middle_friend['user_id'] = user_id
                middle_friend['friends'] = []
                for friend in friends:
                    if friend in middle:
                        middle_friend['friends'].append(friend)
                outfile_middle.write(json.dumps(middle_friend)+'\n')
    exit()
        
    #     type['user'].append({
    #         'label': id,
    #         'user_id': user,
    #         'biz_id': biz,
    #         'categories': categories
    #     })
            
    # with open("output/20_categories", 'w') as outfile:
    #     json.dump(type, outfile)
    # exit()
    
    print(id_sorted)
    matrix = np.zeros((len(id_sorted), len(id_sorted)))
    for i in range(len(id_sorted)):
        for j in range(len(id_sorted)):
            idx_i = id_sorted[i]
            idx_j = id_sorted[j]
            matrix[i][j] = A[idx_i][idx_j]
    np.savetxt("output/covisiting_matrix_20.csv", matrix, fmt='%d', delimiter = ',')

    
    exit()
    
    G = nx.from_numpy_matrix(A)
    connected_components = max(nx.connected_component_subgraphs(G), key = len)
    g = connected_components
    
    component_id = g.nodes()
    print(component_id)
    exit()
    
    matrix = nx.adjacency_matrix(g)
    np.savetxt("output/small.csv", matrix, fmt='%d', delimiter = ",")
    return

def density():
    #read in the raw matrix
    A = np.loadtxt("output/covisiting_matrix")
    n = len(A)
    matrix_new = np.zeros((n,n))
    density_list = []
    size_list = []
    for i in range(n):
        for j in range(n):
            #edge exist when having at least x coreviewed restaurants 
            if int(A[i][j]) > 19:
                matrix_new[i][j] = A[i][j]
    # read from numpy matrix
    A = matrix_new
    G = nx.from_numpy_matrix(A)

    # select the biggest component for the following analysis
    connected_components = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    print("{} connected components found.".format(len(connected_components)))
    g = connected_components[0]
    component_id = g.nodes()
    
    df = pd.DataFrame(index=g.nodes)
    df["degree"] = pd.Series(nx.degree_centrality(g))
    df["betweenness"] = pd.Series(nx.betweenness_centrality(g))
    df["closeness"] = pd.Series(nx.closeness_centrality(g))
    df["eigenvector"] = pd.Series(nx.eigenvector_centrality(g))
    df["clustering"] = pd.Series(nx.clustering(g))
    
    print(df.sort_values("clustering", ascending=False).head(10))
    exit()
   
        
    
    threshold = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
        
    for th in threshold:
        
        node = set()
        edge_num = 0
        for i in range(n):
            for j in range(n):
                #edge exist when having at least x coreviewed restaurants 
                if int(A[i][j]) > th:
                    matrix_new[i][j] = A[i][j]
                    edge_num += 1
                    node.add(i)
                    node.add(j)
        density = edge_num/(n*(n-1))
        density_list.append(density)
        size = len(node)
        size_list.append(size)
    
    # degree distribution plot of the biggest component
    fig1, ax1 = plt.subplots(1,1)
    ax1.scatter(threshold, size_list)
    ax1.plot(threshold, size_list)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Node size")
    ax1.set_title("Node size of network under different threshold")
    
    fig1.savefig('/home/dtao2/Dropbox/size.png')
   
    return
def user_biz():
    # create the user_biz list
    outfile = open("output/user_biz.txt", 'w')
    with open("all_reviews_uc.json", 'r') as f:
        data = json.loads(f.readline())
    for id in user_ids:
        newdata = dict()
        newdata['user_id'] = id
        newdata['biz_id'] = []
        
        for i in range(len(data)):
            user_id = data[i]['user_id']
            business_id = data[i]['business_id']
            if user_id == id:
                newdata['biz_id'].append(business_id)
        outfile.write(json.dumps(newdata))
        outfile.write('\n')
    exit()
    
    outlist = []
    with open("output/user_biz.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            outlist.append(data)
            
    # with open("output/user_biz_list", 'w') as f:
    #     for i in range(len(outlist)):
    #         f.write(outlist[i], f)
            
    print(len(outlist))
    
    return

def biz_categories():
    A = np.loadtxt("output/covisiting_matrix")
    G = nx.from_numpy_matrix(A)

    df_node_degree = pd.DataFrame(list(dict(G.degree()).items()), columns=["node_name", "degree"])
    
    # print(df_node_degree.sort_values("degree", ascending = False).head(10))
    
    # select the biggest component for the following analysis
    connected_components = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    g = connected_components[0]
    component_id = g.nodes()
    
    # assign labels to node
    user_ids = []
    with open("output/component_adjlist.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            user_ids.append(data['user_id'])
    
    labels = {}
    for i in component_id:
        labels[i] = user_ids[i]
    # take closer look at the central node with highest degree and its neighbors 
    central_node_id = user_ids[1721]
    neighbors_id = set()
    for j in g.neighbors(1721):
        neighbors_id.add(user_ids[j])
    
    neighbors_id.add(central_node_id)
    top_users = neighbors_id
    print(top_users)
    
    # user_cat = dict()
    # with open("output/usr_biz.txt", 'r') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         user_id = data['usr_id']
    #         user_cat['user_id'] = user_id
    #         biz_id = data['biz_id']
    #         if user_id in top_users:
    #             for i in range(len(biz_id)):
    #                 biz = biz_id[i]
    #                 cats = set()
    #                 with open("output/biz_catagory.txt", 'r') as f:
    #                     for line in f:
    #                         data = json.laods(line)
    #                         if biz == data['biz_id']:
    #                             for j in data['categories']:
    #                                 cats.add(data['categories'][j])
    #                 cnt = Counter()
    #                 for word in cats:
    #                     cnt[word] += 1
    #                 user_cat['categories_cnt'] = cnt
                            
    
    # find categories of each biz_id for each node in top_ids
    top_biz = set()
    with open("output/user_biz.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            id = data['user_id']
            biz = data['biz_id']
            if id in top_users:
                for i in range(len(biz)):
                    top_biz.add(biz[i])
    print(top_biz)
    
    # get the Counter of categories for each user in top_users
    user_category = dict()
    
    # create biz_category adjacency list
    outfile = open("output/biz_category.txt", 'w')
    with open("all_restaurants_uc", 'r') as f:
        data = json.loads(f.readline())
        for i in range(len(data)):
            biz_categories = dict()
            biz = data[i]['business_id']
            categories = data[i]['categories']
            if biz in top_biz:
                biz_categories['biz'] = biz
                biz_categories['categories'] = categories
                outfile.write(json.dumps(biz_categories))
                outfile.write('\n')
    return

def co_visiting():
    ## create the co_visiting/reviewing network, where nodes are users, edges are number of sharing common restaurant of reviewing, we assume two people have similarity if they visited the same restaurant.
    
    # read from numpy matrix
    A = np.loadtxt("output/covisiting_matrix")
    G = nx.from_numpy_matrix(A)

    # select the biggest component for the following analysis
    connected_components = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    print("{} connected components found.".format(len(connected_components)))
    g = connected_components[0]
    component_id = g.nodes()
    
    df = pd.DataFrame(index=g.nodes)
    df["degree"] = pd.Series(nx.degree_centrality(g))
    df["betweenness"] = pd.Series(nx.betweenness_centrality(g))
    df["closeness"] = pd.Series(nx.closeness_centrality(g))
    df["eigenvector"] = pd.Series(nx.eigenvector_centrality(g))
    df["clustering"] = pd.Series(nx.clustering(g))
    
    print(df.sort_values("betweenness", ascending=False).head(10))
    exit()
    
    df_node_degree = pd.DataFrame(list(dict(g.degree()).items()), columns=["node_name", "degree"])
    print(df_node_degree.sort_values("degree", ascending = False).head(10))
    
    print("radius: {:d}\n".format(nx.radius(g)))
    print("diameter: {:d}\n".format(nx.diameter(g)))
    print("eccentricity: {}\n".format(nx.eccentricity(g)))
    print("center: {}\n".format(nx.center(g)))
    print("periphery: {}\n".format(nx.periphery(g)))
    print("density: {:f}".format(nx.density(g)))
    
    exit()
    
    # assign labels to node
    user_ids = []
    with open("output/component_adjlist.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            user_ids.append(data['user_id'])
    
    labels = {}
    for i in component_id:
        labels[i] = user_ids[i]
    
    # degree distribution plot of the biggest component
    fig1, ax1 = plt.subplots(1,1)
    degree_values = [v for k,v in g.degree()]
    ax1.hist(list(degree_values), bins=list(range(max(degree_values))), log=True)
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Degree Distribution")
    
    fig1.savefig('/home/dtao2/Dropbox/degree.png')
    
    # covisiting  network plot of the biggest component
    fig2, ax2 = plt.subplots(1,1)
    pos=nx.spring_layout(g)
    nx.draw_networkx(
        g, with_labels=False,
        node_size=[x[1]*10 for x in g.degree()],
        pos = pos,
        width=0.5,
        ax=ax2
    )
    
    nx.draw_networkx_labels(g,pos,labels = None, font_size=5)
    
    # nx.draw_networkx_labels(g,pos,labels = labels, font_size=5)
    
    ax2.axis("off")
    ax2.set_title("Covisiting Network")
    fig2.savefig('/home/dtao2/Dropbox/graph.png')

    return

def separate():
    G = nx.Graph()
    with open("output/western.txt", 'r') as f:
        for line in f:
            data = json.loads(line)
            u = data['user_id']
            friend = data['friends']
            for v in friend:
                G.add_edge(u,v)
                

    # select the biggest component for the following analysis
    connected_components = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    print("{} connected components found.".format(len(connected_components)))
    g = connected_components[0]
    
    print('number of nodes: ', g.number_of_nodes())
    print('number of edges: ', g.number_of_edges())
    print('density: ', nx.density(g))
    print('average shortest path: ', nx.average_shortest_path_length(g))
    print('diameter: ', nx.diameter(g))
    
    # draw nodes, edges and labels
    df_node_degree = pd.DataFrame(list(dict(G.degree()).items()), columns=["node_name", "degree"])
    
    degree = df_node_degree.sort_values("degree", ascending=False)['degree']
    sum = 0
    for i in range(len(degree)):
        sum += degree[i]
    average_degree = sum/len(degree)
    print('average degree: ', average_degree) 
    
    # df = pd.DataFrame(index=g.nodes)
    # df["degree"] = pd.Series(nx.degree_centrality(g))
    # df["betweenness"] = pd.Series(nx.betweenness_centrality(g))
    # df["closeness"] = pd.Series(nx.closeness_centrality(g))
    # df["eigenvector"] = pd.Series(nx.eigenvector_centrality(g))
    # df["clustering"] = pd.Series(nx.clustering(g))
    # print(df.sort_values("clustering", ascending=False).head(10))

    
    # friendship network plot of the biggest component
    fig2, ax2 = plt.subplots(1,1)
    nx.draw_networkx(
        g, with_labels=False,
        node_size=[x[1]*10 for x in connected_components[0].degree()],
        pos=nx.spring_layout(connected_components[0]),
        width=0.5,
        ax=ax2
    )
    ax2.axis("off")
    
    # ax2.set_title("c")
    
    fig2.savefig('/home/dtao2/Dropbox/western.png')

def to_json():
    outfile = open('adj_list.txt', 'w')
    cnt = 0
    with open("../user.json", 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['user_id'] not in idx:
                continue

            newdata = dict()
            newdata['user_id'] = data['user_id']
            newdata['friends'] = []
            for friend in data['friends']:
                if friend in idx:
                    newdata['friends'].append(friend)
            outfile.write(json.dumps(newdata))
            outfile.write('\n')

            
if __name__ == "__main__":
    
    user_user()
    
    # small_graph()
    
    # density()
    
    # co_visiting()
    
    # friendship()
    
    # separate()
