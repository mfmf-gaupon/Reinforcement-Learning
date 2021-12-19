import matplotlib.pyplot as plt
import numpy as np

def reward_plotter(history,average=1):
    fig = plt.figure()
    plt.plot(range(1,len(history)+1),history,alpha=0.4,c="orange")
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")

    average_history = []
    size = len(history)
    tmp = 0
    count = 0
    for i,a in enumerate(history):
        tmp += a
        count += 1
        if (i+1)%average==0 or (i+1)==size:
            average_history.append(tmp/count)
            tmp=0
            count=0
    x = np.arange(1,size+1,average)
    if x[-1]!=size:
        np.append(x,size)
    plt.plot(x,average_history)
    plt.savefig("./images/log")
