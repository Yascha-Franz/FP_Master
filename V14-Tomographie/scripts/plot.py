import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as std)
from scipy.optimize import curve_fit

def plotfit(x,y,f,savepath,slice_=slice(0,None),yerr=None, p0=None, save=True, color='k', label='Messwerte'):
    colors = ['k', 'b', 'g', 'r', 'y']
    if (np.size(x[0])>1):
        if yerr==None:
            yerr_=None
        else:
            yerr_=yerr[0]
        if label=='Messwerte':
            label_ = label
        else:
            label_ = label[0]
        param, error = plotfit(x[0],y[0],f,savepath,slice_=slice_,yerr=yerr_, p0=p0, save = False, color=colors[0], label = label_)
        params = [param]
        errors = [error]
        for i in range(1,np.shape(x)[0]):
            if yerr==None:
                yerr_=None
            else:
                yerr_=yerr[i]
            if label=='Messwerte':
                label_ = label
            else:
                label_ = label[i]
            param, error = plotfit(x[i],y[i],f,savepath,slice_=slice_,yerr=yerr_, p0=p0, save = False, color=colors[i], label = label_)
            params = np.append(params, [param], axis = 0)
            errors = np.append(errors, [error], axis = 0)
    else:
        if yerr is None:
            plt.plot(x,y, color=color, linestyle='', marker='.', label =label)
        else:
            plt.errorbar(x,y,yerr=yerr, color=color, linestyle='', marker='x', label =label)
        params, covariance_matrix = curve_fit(f, x[slice_], y[slice_],p0=p0)
        errors = np.sqrt(np.diag(covariance_matrix))
        x_plot = np.linspace(np.min(x[slice_]), np.max(x[slice_]), 1000)
        plt.plot(x_plot, f(x_plot, *params), color=color, linestyle='-', label=f.__name__)
        plt.legend(loc='best')
        plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    if save:
        plt.savefig(savepath)
        plt.clf()
    return params, errors

def polarplot(x,y,f,savepath):
    ax=plt.figure().gca(polar=True)
    ax.plot(x, y, 'kx', label ='Messwerte')
    x_plot = np.linspace(np.min(x), np.max(x), 1000)
    ax.plot(x_plot, f(x_plot), 'b-', label=f.__name__)
    ax.legend(loc='best')
    ax.set_thetamin(np.min(x)*degree)
    ax.set_thetamax(np.max(x)*degree)
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

    plt.savefig(savepath)
    plt.clf()

def Fit(x,a,b):
    return x*a +b
x,y = np.genfromtxt('scripts/data.txt',unpack=True)
plt.xlabel('X')
plt.ylabel('Y')
plotfit(x,y,Fit,'build/plot.pdf')

#gather all the data
counts_1_01 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_1.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_02 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_2.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_03 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_3.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_04 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_4.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_05 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_5.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_06 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_6.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_07 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_7.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_08 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_8.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_09 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_9.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_10 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_10.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_11 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_11.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_1_12 = np.genfromtxt('scripts/data/Würfel_1/Würfel_1_12.Spe', unpack=True, skip_header=12, skip_footer=17)


counts_2_01 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_1.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_02 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_2.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_03 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_3.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_04 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_4.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_05 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_5.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_06 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_6.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_07 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_7.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_08 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_8.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_09 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_9.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_10 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_10.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_11 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_11.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_2_12 = np.genfromtxt('scripts/data/Würfel_2/Würfel_2_12.Spe', unpack=True, skip_header=12, skip_footer=17)


counts_3_01 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_1.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_02 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_2.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_03 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_3.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_04 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_4.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_05 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_5_2.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_06 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_6.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_07 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_7.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_08 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_8.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_09 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_9.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_10 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_10.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_11 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_11.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_3_12 = np.genfromtxt('scripts/data/Würfel_3/Würfel_3_12.Spe', unpack=True, skip_header=12, skip_footer=17)


counts_4_01 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_1.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_02 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_2.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_03 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_3.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_04 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_4.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_05 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_5.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_06 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_6.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_07 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_7.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_08 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_8.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_09 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_9.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_10 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_10.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_11 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_11.Spe', unpack=True, skip_header=12, skip_footer=17)
counts_4_12 = np.genfromtxt('scripts/data/Würfel_4/Würfel_4_12.Spe', unpack=True, skip_header=12, skip_footer=17)

#put the data in managable arrays
counts_1 = np.array([counts_1_01, counts_1_02, counts_1_03, counts_1_04, counts_1_05, counts_1_06, counts_1_07, counts_1_08, counts_1_09, counts_1_10, counts_1_11, counts_1_12])

counts_2 = np.array([counts_2_01, counts_2_02, counts_2_03, counts_2_04, counts_2_05, counts_2_06, counts_2_07, counts_2_08, counts_2_09, counts_2_10, counts_2_11, counts_2_12])

counts_3 = np.array([counts_3_01, counts_3_02, counts_3_03, counts_3_04, counts_3_05, counts_3_06, counts_3_07, counts_3_08, counts_3_09, counts_3_10, counts_3_11, counts_3_12])

counts_4 = np.array([counts_4_01, counts_4_02, counts_4_03, counts_4_04, counts_4_05, counts_4_06, counts_4_07, counts_4_08, counts_4_09, counts_4_10, counts_4_11, counts_4_12])

#select the photopeak
x_min = 143
x_max = 171

x_scale_total = np.linspace(100, 180-1, 80)
plt.step(x_scale_total, counts_4[7, 100:180])
#x_scale_total = np.linspace(0, np.shape(counts_1)[1]-1, np.shape(counts_1)[1])
#plt.step(x_scale_total, counts_1[0])

x_scale = np.linspace(x_min, x_max-1, x_max-x_min)
plt.step(x_scale, counts_4[7, x_min:x_max], color='red',label='integrierter bereich')


plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/photopeak.pdf')
plt.clf()

#integrate the photopeak
counts_1_integrated = np.sum(counts_1[:,x_min:x_max], axis=1)
counts_2_integrated = np.sum(counts_2[:,x_min:x_max], axis=1)
counts_3_integrated = np.sum(counts_3[:,x_min:x_max], axis=1)
counts_4_integrated = np.sum(counts_4[:,x_min:x_max], axis=1)

#for errors
counts_1_unp = unp.uarray(counts_1_integrated, np.sqrt(counts_1_integrated))
counts_2_unp = unp.uarray(counts_2_integrated, np.sqrt(counts_2_integrated))
counts_3_unp = unp.uarray(counts_3_integrated, np.sqrt(counts_3_integrated))
counts_4_unp = unp.uarray(counts_4_integrated, np.sqrt(counts_4_integrated))

#check statistic
#print(counts_1_integrated)
#print(counts_2_integrated)
#print(counts_3_integrated) #5 and 11 still need to be halved because their time was double the others
#print(counts_4_integrated)

#fix the more time on 3_5 and 3_11
counts_3_unp[4] /= 2
counts_3_unp[10] /= 2

#define the array A
w = np.sqrt(2)
A = np.array([[1,1,1,0,0,0,0,0,0],
              [0,0,0,1,1,1,0,0,0],
              [0,0,0,0,0,0,1,1,1],
              [0,w,0,w,0,0,0,0,0],
              [0,0,w,0,w,0,w,0,0],
              [0,0,0,0,0,w,0,w,0],
              [1,0,0,1,0,0,1,0,0],
              [0,1,0,0,1,0,0,1,0],
              [0,0,1,0,0,1,0,0,1],
              [0,0,0,w,0,0,0,w,0],
              [w,0,0,0,w,0,0,0,w],
              [0,w,0,0,0,w,0,0,0]])


#calculate the inverse of A.T x A
A_inverse = np.linalg.inv(A.T @ A)

#calculate the normalized fluxes (doing it this way around takes care of the minus in the exponential)
I_2_unp = unp.log(counts_1_unp/counts_2_unp)
I_3_unp = unp.log(counts_1_unp/counts_3_unp)
I_4_unp = unp.log(counts_1_unp/counts_4_unp)

#revert back to normal arrays, cause matmul doesn't work with unumpy
I_2 = noms(I_2_unp)
I_2_err = std(I_2_unp)
I_3 = noms(I_3_unp)
I_3_err = std(I_3_unp)
I_4 = noms(I_4_unp)
I_4_err = std(I_4_unp)

#calculate the absorptioncoefficients and their errors
µ_2 = A_inverse @ A.T @ I_2
µ_2_err = np.sqrt(np.diag(np.linalg.inv(A.T @ np.diag(1/I_2_err**2) @ A)))
µ_3 = A_inverse @ A.T @ I_3
µ_3_err = np.sqrt(np.diag(np.linalg.inv(A.T @ np.diag(1/I_3_err**2) @ A)))
µ_4 = A_inverse @ A.T @ I_4
µ_4_err = np.sqrt(np.diag(np.linalg.inv(A.T @ np.diag(1/I_4_err**2) @ A)))

print('Absorptionskoeffizienten in 1/cm:')
print('Würfel 2: ', µ_2, ' +- ', µ_2_err)
print('-------------------------------')
print('Würfel 3: ', µ_3, ' +- ', µ_3_err)
print('-------------------------------')
print('Würfel 4: ', µ_4, ' +- ', µ_4_err)
print('-------------------------------')
print('')
print('-------------------------------')
print('')
print('Gemittelte Absorptionskoeffizienten für Würfel 2 & 3')
print('Würfel 2: ', np.mean(µ_2), ' +- ', np.std(µ_2))
print('-------------------------------')
print('Würfel 3: ', np.mean(µ_3), ' +- ', np.std(µ_3))
print('-------------------------------')