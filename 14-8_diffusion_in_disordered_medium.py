#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto, August 2014.

from Tkinter import *
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import sys
import time

class Percolation:

    def __init__(self, L=61):
        self.sub = None
        self.L = L
        self.count = 0
        
    def perc_cluster(self, p=0.7):
        if p > 1 or p < 0:
            raise ValueError("site occupation probability must be 0 <= p <= 1")
        self.p = p
        self.lattice = np.zeros([self.L+2, self.L+2], dtype=int)
        self.lattice[:1, :] = self.lattice[:, :1] = -1
        self.lattice[self.L+1:, :] = self.lattice[:, self.L+1:] = -1
        self.center = int(self.L/2) + 1
        self.lattice[self.center, self.center] = 1
        nextseed = [(self.center, self.center)]
        if self.sub is None or not self.sub.winfo_exists():
            lattice = self.lattice
            rn = np.random.random
            ne = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            nnsite = set([(self.center+nx, self.center+ny) for nx, ny in ne])
            percolate = False
            l = set([])
            while len(nnsite) != 0 and percolate == False:
                nextseed = []
                for nn in nnsite:
                    if rn() < p:
                        lattice[nn] = 1
                        nextseed.append(nn)
                    else: lattice[nn] = -1
                nnsite = set([])
                for i, j in nextseed:
                    nnsite = nnsite | set([(i+nx, j+ny) for nx, ny in ne
                                if lattice[i+nx, j+ny] == 0])
                    if i == 1:      l = l | set(['top'])
                    if i == self.L: l = l | set(['bottom'])
                    if j == 1:      l = l | set(['left'])
                    if j == self.L: l = l | set(['right'])
                
                if ('top' in l and 'bottom' in l) or \
                   ('right' in l and 'left' in l):
                    percolate = True
                
            if len(nnsite) == 0 and self.count < 30:
                self.count += 1
                self.perc_cluster(self.p)
            else:
                self.count = 0
                self.lattice = lattice[1:-1, 1:-1]
                return self.lattice
        
    def draw_canvas(self, rect, L):
        default_size = 640 # default size of canvas
        r = int(default_size/(2*L))
        fig_size = 2*r*L
        margin = 10
        self.sub = Toplevel()
        
        self.sub.title('invasion percolation')
        self.canvas = Canvas(self.sub, width=fig_size+2*margin,
                    height=fig_size+2*margin)
        self.canvas.create_rectangle(margin, margin,
                    fig_size+margin, fig_size+margin,
                    outline='black', fill='white')
        self.canvas.pack()
        
        c = self.canvas.create_rectangle
        
        site = np.where(rect==1)
        for m, n in zip(site[0], site[1]):
            c(2*m*r+margin, 2*n*r+margin,
                        2*(m+1)*r+margin, 2*(n+1)*r+margin,
                        outline='black', fill='black')
        self.r = r
        self.margin = margin

class Ant():
    
    def rw_d2(self, center, lattice, r, margin, f, N=1000, view=True):
        
        R_2 = []
        x, y = center, center
        per_lattice = lattice == 1
        rn = np.random.rand
        if view:
            oval = f.create_oval
            delete = f.delete
            ant = oval(2*x*r+margin, 2*x*r+margin,
                            2*(x+1)*r+margin, 2*(x+1)*r+margin,
                            outline='white', fill='red')
        for n in xrange(N):
            p = rn()*4
            if p < 1:   d = (0, 1)
            elif p < 2: d = (0, -1)
            elif p < 3: d = (1, 0)
            else:       d = (-1, 0)
            
            _x, _y = x+d[0], y+d[1]
            
            # 進行方向が占有サイトのときのみ進める
            try:
                if per_lattice[_x][_y]:
                    x, y = _x, _y
                    if view:
                        delete(ant)
                        ant = oval(2*x*r+margin, 2*y*r+margin,
                                2*(x+1)*r+margin, 2*(y+1)*r+margin,
                                outline='white', fill='red')
                        f.update()
                        time.sleep(0.02)
            except IndexError:
                _x, _y = x, y
            R_2.append((x-center)**2 + (y-center)**2)
        t = xrange(1, N+1)
        return t, R_2

class TopWindow:
    
    def quit(self):
        self.root.destroy()
        sys.exit()
        
    def show_window(self, title="title", *args):
        self.root = Tk()
        self.root.title(title)
        frames = []
        for i, arg in enumerate(args):
            frames.append(Frame(self.root, padx=5, pady=5))
            for k, v in arg:
                Button(frames[i],text=k,command=v).pack(expand=YES, fill='x')
            frames[i].pack(fill='x')
        f = Frame(self.root, padx=5, pady=5)
        Button(f,text='quit',command=self.quit).pack(expand=YES, fill='x')
        f.pack(fill='x')
        self.root.mainloop()

class Common():

    def plot_graph(self, x_data, y_data, x_labels, y_labels,
                    xscale, yscale, aspect):
        """ Plot the graph about y_data for each x_data.
        """
        d = len(y_data)
        if not len(x_data) == len(y_data) == len(x_labels) == len(y_labels)\
               == len(xscale) == len(yscale) == len(aspect):
            raise ValueError("Arguments must have the same dimension.")
        if d == 0:
            raise ValueError("At least one data for plot.")
        if d > 9:
            raise ValueError("""So much data for plot in one figure.
                                Please divide two or more data sets.""")

        fig = plt.figure(figsize=(9, 8))
        subplot_positioning = ['11','21','22','22','32','32','33','33','33']
        axes = []
        for n in range(d):
            lmn = int(subplot_positioning[d-1] + str(n+1))
            axes.append(fig.add_subplot(lmn))

        for i, ax in enumerate(axes):
            ymin, ymax = min(y_data[i]), max(y_data[i])
            ax.set_aspect(aspect[i])
            ax.set_xscale(xscale[i])
            ax.set_yscale(yscale[i])
            ax.set_xlabel(x_labels[i], fontsize=16)
            ax.set_ylabel(y_labels[i], fontsize=16)
            ax.set_ymargin(0.05)
            ax.plot(x_data[i], y_data[i], 'o-')

        fig.subplots_adjust(wspace=0.2, hspace=0.5)
        fig.tight_layout()
        plt.show()
        
    def fitting(self, x, y, parameter0, fit_func):
        
        parameter = parameter0
        result = optimize.leastsq(fit_func, parameter0, args=(x, y))
        for i in range(len(parameter)):
            parameter[i] = result[0][i]
        
        return parameter

class Main():

    def __init__(self):
        self.L = 61
        self.p = 0.7
        self.top = TopWindow()
        self.per = Percolation(self.L)
        self.ant = Ant()
        self.common = Common()
        self.count = 1
        run = (('percolation cluster', self.pushed),)
        run2 = (('ant_walk', self.ant_walk),
                        ('calculate R_2', self.calculate_R_2),
                        (r'caluculate D_s(p)', self.fit))
        save = (('save canvas to sample.eps', self.pr),)
        self.top.show_window("Ant Walk", run, run2, save)
    
    def pr(self):
        d = self.per.canvas.postscript(file="figure_%d.eps" % self.count)
        print "saved the figure to a eps file"
        self.count += 1

    def pushed(self):
        self.per.perc_cluster(self.p)
        self.per.draw_canvas(self.per.lattice, self.L)
    
    def ant_walk(self):
        if self.per.sub == None or not self.per.sub.winfo_exists():
            self.per.perc_cluster(self.p)
            self.per.draw_canvas(self.per.lattice, self.L)
        t, R_2 = self.ant.rw_d2(self.per.center, self.per.lattice,
                        self.per.r, self.per.margin, self.per.canvas)

    def calculate_R_2(self):
        trial = 1000
        N = 5000
        R_2 = []
    
        for i in range(trial):
            self.per.perc_cluster(self.p)
            t, R_2_ = self.ant.rw_d2(self.per.center, self.per.lattice,
                            0, 0, 0, N, view=False)
            R_2.append(R_2_)
        R_2 = np.array(R_2).reshape(trial, N)
        self.ave_R_2 = np.sum(R_2, axis=0)/float(trial)
        self.t = t
        self.common.plot_graph([self.t], [self.ave_R_2], [r'$t$'], 
                        [r'$\langle R^{2}(t) \rangle$'], 
                        ['linear'], ['linear'], ['auto'])
    
    def fit(self, view=True):
        
        def fit_func(parameter0, t, R_2):
            a = parameter0[0]
            b = parameter0[1]
            residual = R_2 - (a*t + b)
            return residual
                
        def fitted(t, a, b):
            return a*t + b
        
        cut_from = int(raw_input("from ? (t) >>> "))
        cut_to = int(raw_input("to ? (t) >>> "))
        parameter0 = [0.5, 1.]
        cut_t =  np.array(list(self.t)[cut_from:cut_to])
        result = self.common.fitting(cut_t, 
                        np.array(self.ave_R_2[cut_from:cut_to]),
                        parameter0, fit_func)
        a = result[0]
        D = a/4.
        b = result[1]
        print D
        if view:
            fig = plt.figure('Diffusion coefficient')
            ax = fig.add_subplot(111)
            ax.plot(self.t, self.ave_R_2, '-o',
                            label=r"$\langle R^{2}(t) \rangle$")
            ax.plot(cut_t, fitted(cut_t, a, b), lw=2,
                            label=r"fit func: $D_{s}(p) = %f$" % D)
            ax.set_xlabel(r'$t$', fontsize=16)
            ax.set_ylabel(r'$\langle R^{2}(t) \rangle$', fontsize=16)
            ax.set_xscale('linear')
            ax.set_yscale('linear')
            ax.set_ymargin(0.05)
            fig.tight_layout()
            plt.legend(loc='best')
            plt.show()
        return D
        
if __name__ == '__main__':
    
    Main()
    
