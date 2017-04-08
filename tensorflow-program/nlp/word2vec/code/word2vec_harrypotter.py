#coding=GBK
'''
Created on 2017年4月5日

@author: Scorpio.Lu
'''
import collections
import re  
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
'''
读取源文件,并转为list输出
@param filename:文件名
@return: list of words
'''
def read_file(filename):
    f=open(filename,'r')
    file_read=f.read()
    words_=re.sub("[^a-zA-Z]+", " ",file_read).lower() #正则匹配,只留下单词，且大写改小写
    words=list(words_.split())  #length of words:1121985
    return words
    
words=read_file('哈利波特1-7英文原版.txt')


vocabulary_size=2000  #预定义频繁单词库的长度
count=[['UNK',-1]]    #初始化单词频数统计集合

'''
1.给@param “words”中出现过的单词做频数统计，取top 1999频数的单词放入dictionary中，以便快速查询。
2.给哈利波特这本“单词库”@param“words”编码，出现在top 1999之外的单词，统一令其为“UNK”（未知），编号为0，并统计这些单词的数量。
@return: 哈利波特这本书的编码data，每个单词的频数统计count，词汇表dictionary及其反转形式reverse_dictionary
'''
def build_dataset(words):
    counter=collections.Counter(words).most_common(vocabulary_size-1) #length of all counter:22159 取top1999频数的单词作为vocabulary，其他的作为unknown
    count.extend(counter)
    #搭建dictionary
    dictionary={}
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data=[]
    #全部单词转为编号
    #先判断这个单词是否出现在dictionary，如果是，就转成编号，如果不是，则转为编号0（代表UNK）
    unk_count=0
    for word in words:
        if word in dictionary:
            index=dictionary[word]
        else:
            index=0
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary

data,count,dictionary,reverse_dictionary=build_dataset(words)   
del words #删除原始单词列表，节约内存



data_index=0


'''
采用Skip-Gram模式
生成word2vec训练样本
@param batch_size:每个批次训练多少样本
@param num_skips: 为每个单词生成多少样本（本次实验是2个），batch_size必须是num_skips的整数倍,这样可以确保由一个目标词汇生成的样本在同一个批次中。
@param skip_window:单词最远可以联系的距离（本次实验设为1，即目标单词只能和相邻的两个单词生成样本），2*skip_window>=num_skips
'''
def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size%num_skips==0
    assert num_skips<=2*skip_window
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1   #入队长度
    buffer=collections.deque(maxlen=span)
    
    for _ in range(span):  #双向队列填入初始值
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)  
        
    for i in range(batch_size//num_skips):  #第一次循环，i表示第几次入双向队列deque
        for j in range(span):  #内部循环，处理deque
            if j>skip_window:
                batch[i*num_skips+j-1]=buffer[skip_window]
                labels[i*num_skips+j-1,0]=buffer[j]
            elif j==skip_window:
                continue
            else:
                batch[i*num_skips+j]=buffer[skip_window]
                labels[i*num_skips+j,0]=buffer[j]
        buffer.append(data[data_index])  #入队一个单词，出队一个单词
        data_index=(data_index+1)%len(data)
    return batch,labels    


#开始训练
batch_size=128   
embedding_size=128
skip_window=1
num_skips=2
num_sampled=64  #训练时用来做负样本的噪声单词的数量
#验证数据
valid_size=16 #抽取的验证单词数
valid_window=100 #验证单词只从频数最高的100个单词中抽取
valid_examples=np.random.choice(valid_window,valid_size,replace=False)#不重复在0——10l里取16个


graph=tf.Graph()
with graph.as_default():
    train_inputs=tf.placeholder(tf.int32, shape=[batch_size])
    train_labels=tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
    embeddings=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size], -1, 1))   #初始化embedding vector
    embed=tf.nn.embedding_lookup(embeddings, train_inputs) 
    
    #用NCE loss作为优化训练的目标 
    nce_weights=tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size], stddev=1.0/np.math.sqrt(embedding_size)))
    nce_bias=tf.Variable(tf.zeros([vocabulary_size]))
    loss=tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_bias, embed, train_labels, num_sampled, num_classes=vocabulary_size))
    optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
    normalized_embeddings=embeddings/norm   #除以其L2范数后得到标准化后的normalized_embeddings
    
    valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)    #如果输入的是64，那么对应的embedding是normalized_embeddings第64行的vector
    similarity=tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)   #计算验证单词的嵌入向量与词汇表中所有单词的相似性
    
    init=tf.global_variables_initializer()
    
num_steps=1000001
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")
    avg_loss=0
    for step in range(num_steps):
        batch_inputs,batch_labels=generate_batch(batch_size, num_skips, skip_window)  #产生批次训练样本
        feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}   #赋值
        _,loss_val=session.run([optimizer,loss],feed_dict=feed_dict)
        avg_loss+=loss_val
        if step % 2000 ==0:
            if step>0:
                avg_loss/=2000
            print("Avg loss at step ",step,": ",avg_loss)
            avg_loss=0
        if step%10000==0:
            sim=similarity.eval()
            for i in range(valid_size):
                valid_word=reverse_dictionary[valid_examples[i]]  #得到验证单词
                top_k=8  
                nearest=(-sim[i,:]).argsort()[1:top_k+1]     #每一个valid_example相似度最高的top-k个单词
                log_str="Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word=reverse_dictionary[nearest[k]]
                    log_str="%s %s," %(log_str,close_word)
                print(log_str)
    final_embedding=normalized_embeddings.eval()
'''
可视化Word2Vec散点图并保存
'''
def plot_with_labels(low_dim_embs,labels,filename):
    assert low_dim_embs.shape[0]>=len(labels),"more labels than embedding"
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
    plt.savefig(filename)

'''
tsne实现降维，将原始的128维的嵌入向量降到2维
'''

tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
plot_number=150
low_dim_embs=tsne.fit_transform(final_embedding[:plot_number,:])
labels=[reverse_dictionary[i] for i in range(plot_number)]
plot_with_labels(low_dim_embs, labels, './plot.png')

       

    