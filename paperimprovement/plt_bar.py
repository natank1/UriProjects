import matplotlib.pyplot as plt
from our_scores import build_socre,pair_wise_process
p00 = [-2,-1,-0.5,0.5,1,2]
def plt_bar_o(cm):
    scores=[]
    for p in p00:
        scores.append(pair_wise_process(cm,p,p))
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    xx = plt.plot(p00, scores, color='maroon'              )

    plt.xlabel("P ")
    plt.ylabel("Scores")
    plt.title("4 Classes Naive Bayes Pairwise ")
    plt.legend(['All', 'Harmonic', 'Algebr'])
    plt.show()
    return


