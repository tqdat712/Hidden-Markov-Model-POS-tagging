from cmath import inf
import numpy as np
from lib.strLib import cleanText, normalizedText
from lib.easyX import easyX
import matplotlib.pyplot as plt 


FILENAMES = [["./data/train_preprocessed.csv", "train"],
            ["./data/test_preprocessed.csv", "test"],
            ["./data/dev_preprocessed.csv", "dev"]]

FILE_TRAIN = "./data/train_preprocessed.csv"
FILE_VOCAB = "./data/vocab.ezx"

OUTPREFIX = "./results/HMM_single_probability_"
OUTLISTS = "./results/HMM_single_probability_meta.ezx"


ezx = easyX()
vocab = ezx.load(FILE_VOCAB)

id2word = vocab["id2word"]
word2id = vocab["word2id"]
id2tag  = vocab["id2tag"]
tag2id  = vocab["tag2id"]



def get_data(fname):
    fheader = open(fname, 'r')
    Lines = fheader.readlines()

    samples = []
    for linenum, line in enumerate(Lines):
        (words, tags) = line.split('\t')
        wordlist = words.strip().split(' ')
        taglist = tags.strip().split(' ')

        try:
            X = [ word2id[w] for w in wordlist ]
            Y = [ tag2id[t]  for t in taglist ]
        except:
            print(linenum, line, wordlist)
            exit(1)
        samples.append((X,Y))

    fheader.close()
    return samples

def train(samples, k):
    # samples: a list of samples; each sample: a seq of words, a seq of POS tags
    # k: add-k smoothing
    # Note: supervised learning for BN is simply counting

    num_words = len(word2id)
    num_tags  = len(tag2id)

    P_init = np.ones(num_tags) * k
    P_transition = np.ones( (num_tags, num_tags) ) * k
    P_emission = np.ones( (num_tags, num_words )) * k


    # TODO: your code here
    for (words, tags) in samples:
        P_init[tags[0]] += 1
        for i in range(len(tags) - 1):
            P_transition[tags[i], tags[i+1]] += 1
        for i in range(len(words)):
            P_emission[tags[i], words[i]] += 1

    P_init /= np.sum(P_init)
    P_transition /= np.sum(P_transition, axis=1, keepdims=True)
    P_emission /= np.sum(P_emission, axis=1, keepdims=True)
    


    return (P_init, P_transition, P_emission)

def inf_logprob(samples, HMM):

    (P_init, P_transition, P_emission) = HMM

    sum_log_prob = 0
    cnt = 0

    for (words, tags) in samples:
        # continue
        # TODO: your code here
        
        sum_log_prob = sum_log_prob + np.log(P_init[tags[0]]*P_emission[tags[0]][words[0]])
        for i in range(1,len(words)):
                sum_log_prob = sum_log_prob + np.log(P_emission[tags[i]][words[i]]*P_transition[tags[i-1]][[tags[i]]]) 

    return sum_log_prob

def MAP(samples, HMM):

    (P_init, P_transition, P_emission) = HMM
    
    num_tags  = len(tag2id)
    
    forward = np.ones((len(samples), num_tags))
    backward = np.ones((len(samples), num_tags))
    
    guess = np.ones(len(samples))
    for i in range(1, len(samples)):
        for j in range(num_tags):
            
            forward[i,j] = np.sum(np.reshape(P_init, (-1,1))*np.transpose(forward[i-1,:])*P_transition[:,j]*P_emission[:,samples[i-1]])   
            
    for i in range(len(samples)-2,-1,-1):
        for j in range(num_tags):
            backward[i,j] = np.sum(np.transpose(backward[i+1,:])*np.transpose(P_transition[j,:])*P_emission[:,samples[i+1]])
    
    for i in range(len(guess)):
        guess[i] = np.argmax(forward[i,:]*backward[i,:]*np.transpose(P_emission[:,samples[i]]))

    
    result = []

    for i in guess.astype(int):
        result.append(id2tag[i])
    return result

if __name__ == '__main__':

    samples = get_data(FILE_TRAIN)
    val_data = get_data(FILENAMES[2][0])
    test_data = get_data(FILENAMES[1][0])

    
    HMM = train(samples, 0.00001)

    test_sent1   = 'I LIKE TO EAT'
    test_sent2   = 'NO ONE LIKES MOVIES'
    
    X = [ word2id[w] for w in test_sent1.split() ]
    Y = [ word2id[w] for w in test_sent2.split() ]
        
    print(MAP(X, HMM))
    print(MAP(Y, HMM))
    # k_s = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    
#     # result_list = []
#     # for k in k_s:
#     #     HMM = train(samples, k)
#     #     tags_count = 0
#     #     word_accuracy = 0
#     #     sentence_accuracy = 0
        
#     #     for (words, tags) in val_data:
            
            
#     #         predict = MAP(words,HMM)
#     #         word = 0
#     #         sentence = 1
#     #         for i, tag in enumerate(tags):
#     #             tags_count += 1
#     #             if tag2id[predict[i]] == tag:
#     #                 word_accuracy += 1
            
#     #             if word < len(tags):
#     #                 sentence = 0 
                
#     #             sentence_accuracy += sentence
#     #     print(word_accuracy, tags_count)
#     #     word_accuracy /= tags_count
#     #     sentence_accuracy /= len(val_data)
        
#     #     result_list.append(word_accuracy, sentence_accuracy)
        
    
#     # for i in result_list:
#     #     if i == inf or i == -inf:
#     #         i = 0
#     # print('return ',result_list )
#     # print(np.log10(k_s))




#     # fig, ax = plt.subplots()
#     # # ax.axhline(y=result_list[0])
#     # ax.plot(np.log10(k_s),result_list,'o')
#     # plt.show()
    

#     # HMM = train(samples, 0.00001)
#     (P_init, P_transition, P_emission) = HMM
#     ##########################################
#     tags_count = 0
#     word_accuracy = 0
#     sentence_accuracy = 0
        
#     for (words, tags) in test_data:
        
        
#         predict = MAP(words,HMM)
        
        
#         for i, tag in enumerate(tags):
#             tags_count += 1
#             if tag2id[predict[i]] == tag:
#                 word_accuracy += 1
#         # print([tag2id[predict[i]] for i in range(len(tags))], tags, [tag2id[predict[i]] for i in range(len(tags))] == tags)
#         if [tag2id[predict[i]] for i in range(len(tags))] == tags:
#             sentence_accuracy += 1
          
            
#     # print(sentence_accuracy)
#     # print(word_accuracy, tags_count)
#     word_accuracy /= tags_count
#     sentence_accuracy /= len(val_data)
        
#     print(word_accuracy, sentence_accuracy)    
# ###########################################################
     
    
#     tags_count = 0
#     word_accuracy = 0
#     sentence_accuracy = 0
    

#     for (words, tags) in test_data:
#         predict = [-1] * len(words)
#         for j,_ in enumerate(words):
#             # max = np.max(P_emission[:,_])
#             # max_id = 0
#             # for i in range(len(P_emission[:,_])):
#             #     # print(P_emission[i,j])
#             #     if P_emission[i,_] == max:
#             #         # print(P_emission[i,_], max, i, 'dddddd', j)
#             #         max = P_emission[i,_]
#             #         max_id = i                
            
#             predict[j] = int(np.argmax(P_emission[:,_]))
        
#         for i, tag in enumerate(tags):
#             tags_count += 1
#             if predict[i] == tag:
#                 word_accuracy += 1
#         # print([tag2id[predict[i]] for i in range(len(tags))], tags, [tag2id[predict[i]] for i in range(len(tags))] == tags)
        
#         if predict == tags:
#             sentence_accuracy += 1
#     word_accuracy /= tags_count
#     sentence_accuracy /= len(test_data)
#     print(word_accuracy, sentence_accuracy)    