# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

default_sizes = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
from para_data import para_sizes,Mflops_parasize_blocked,Mflops_parasize_rb,Mflops_parasize_copy,Mflops_parasize_autovect,Mflops_parasize_genvect
from data1 import test_sizes,Mflops_julia,Mflops_genvect,Mflops_autovect,Mflops_basic,Mflops_blocked,Mflops_col,Mflops_copy,Mflops_naive,Mflops_rb,Mflops_row,Mflops_blas

plt.rcParams['figure.figsize'] = (12.0, 6.0) # Set the figure_size size

if __name__ == '__main__':

    nsamples = 20
    colors = ['magenta','red','tan','blue','indigo','turquoise','yellow','greenyellow','chocolate','teal','darkblue']

    def re_order(mflops_data):
        for i in range(nsamples):
            tmp = [None] * len(test_sizes)
            for j in range(len(test_sizes)):
                size_n = test_sizes[j]
                order = int((size_n-400)/50)
                tmp[order] = mflops_data[i][j]

            mflops_data[i] = tmp

        return mflops_data

    def draw_proplot(Mflops_data,i_size):
        Mflops = re_order(Mflops_data)
        samples = []
        for i in range(nsamples):
            samples.append(Mflops[i][i_size])

        box = np.array(samples)
        mean = np.mean(box)
        std = np.std(box,ddof=1)

        interval = stats.t.interval(0.95, nsamples - 1, mean, std / (nsamples ** 0.5))  #

        CI = (interval[1] - interval[0]) / 2;

        print("samples:",samples)
        print("mean:",mean)
        print("std:",std)
        print("CI:",CI)

        sparams = (mean,std)
        stats.probplot(box, dist=stats.norm, sparams=sparams, plot=plt)
        plt.show()


    def draw_interval(Mflops_data,color,linestyle):
        Mflops = re_order(Mflops_data)
        y_arr = []
        for i in range(nsamples):
            y_mflops=[]
            x_size=[]
            for i_size in range(len(default_sizes)):
                y_mflops.append(Mflops[i][i_size])
                x_size.append(default_sizes[i_size])

            # plt.plot(x_size, y_mflops, c=colors[i])
            y_arr.append(y_mflops)

        y_mean = []
        y_ci_lower = []
        y_ci_upper = []
        y_err = []
        for k in range(len(x_size)):
            pts = []
            for i in range(nsamples):
                pts.append(y_arr[i][k])

            # compute CI interval using T-distribution algorithm
            points = np.array(pts)
            mean = np.mean(points)
            std = np.std(points)
            interval = stats.t.interval(0.95, nsamples - 1, mean, std/(nsamples**0.5))   #

            y_ci_lower.append(interval[0])
            y_ci_upper.append(interval[1])
            y_err.append((interval[1]-interval[0])/2)
            y_mean.append(mean)

        plt.errorbar(x_size, y_mean, yerr=y_err, ecolor=color, color=color, linestyle=linestyle)


    def draw_box(data1,data2,i_size,name1,name2):
        box_data = []
        samples=[]
        for i in range(nsamples):
            samples.append(data1[i][i_size])

        box_data.append(samples)

        samples=[]
        for i in range(nsamples):
            samples.append(data2[i][i_size])

        box_data.append(samples)

        fig,ax = plt.subplots()
        ax.boxplot(box_data,positions=[1,2])
        ax.set_xticklabels([name1, name2])
        plt.ylabel(u'seconds')

        plt.show()



    def draw_para_line(Mflops_parasize):
        y_mean = []
        x_size = []
        for i in range(len(para_sizes)):
            mean = np.mean(np.array(Mflops_parasize[i]))
            x_size.append(para_sizes[i])
            y_mean.append(mean)

        # plt.ylim(1000,13000)
        plt.xlabel(u'BI,BJ,BK value')
        plt.ylabel(u'MFlops/s')
        plt.title("rb parameter performance")
        plt.plot(x_size, y_mean, c='b')
        plt.show()


    def draw_errorbar_line():
        # plt.ylim(0,80000)
        # plt.ylim(4000,5000)
        # plt.ylim(0,4000)
        # plt.ylim(0,9000)
        plt.ylim(0.1015,0.1030)

        plt.xlim(596,604)
        # plt.xticks(np.arange(len(default_sizes)),default_sizes)
        draw_interval(Mflops_julia,colors[0],'-')
        draw_interval(Mflops_genvect,colors[1],'--'),
        draw_interval(Mflops_autovect,colors[2],'-.')
        draw_interval(Mflops_rb,colors[3],':')
        draw_interval(Mflops_copy,colors[4],'-')
        draw_interval(Mflops_blocked,colors[5],'--')
        draw_interval(Mflops_row,colors[6],'-.')
        draw_interval(Mflops_col,colors[7],':'),
        draw_interval(Mflops_naive,colors[8],'-')
        draw_interval(Mflops_basic,colors[9],'--')
        draw_interval(Mflops_blas,colors[10],'-.')

        plt.legend(['julia', 'genvect', 'autovect', 'rb', 'copy', 'blocked', 'row', 'col', 'naive', 'basic','blas'],loc='upper left')
        plt.title(u'benchmark mean/error bar')
        plt.xlabel(u'Matrix size N*N')
        plt.ylabel(u'seconds')

        plt.show()


    def computre_hypo(data1,data2):
        # Mflops = re_order(Mflops_data)
        for i_size in [2,4,6,8,10]:
            size = default_sizes[i_size]
            samples = []
            for i in range(nsamples):
                samples.append(data1[i][i_size])

            box = np.array(samples)
            mu0 = np.mean(box)

            samples = []
            for i in range(nsamples):
                samples.append(data2[i][i_size])

            box = np.array(samples)
            mean = np.mean(box)
            std = np.std(box, ddof=1)
            interval = stats.t.interval(0.95, nsamples - 1, mean, std / (nsamples ** 0.5))  #
            CI = (interval[1] - interval[0]) / 2;
            t0 = (mean-mu0) / (std/4.4721)

            # print("samples:",samples)
            print("size:%d,mu0:%f,mean:%f,std:%f,t0:%f,CI:%f"%(size,mu0,mean,std,t0,CI))



    #  Test optimal parameters
    # draw_para_line(Mflops_parasize_rb)

    # Test performance
    # draw_errorbar_line()

    draw_box(Mflops_copy,Mflops_blocked,4,'benchmark_copy','benchmark_blocked')

    # Print probability plot
    # draw_proplot(Mflops_copy, 4)

    # computre_hypo(Mflops_blocked, Mflops_copy)

