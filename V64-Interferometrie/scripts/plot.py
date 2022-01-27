import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as cst
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as std)

def plotfit(x,y,f,savepath,slice_=slice(0,None),yerr=None, p0=None, save=True, color='k', label= None, fitlabel = None):
    colors = ['b', 'g', 'r', 'y']
    if (np.size(x[0])>1):
        if yerr==None:
            yerr_=None
        else:
            yerr_=yerr[0]
        if label is None:
            label_ = 'Messwerte'
        else:
            if type(label) is str:
                label_ = label
            else:
                label_ = label[0]
        param, error = plotfit(x[0],y[0],f,savepath,slice_=slice_,yerr=yerr_, p0=p0, save = False, color=colors[0], label = label_, fitlabel = fitlabel[0])
        params = [param]
        errors = [error]
        for i in range(1,np.shape(x)[0]):
            if yerr==None:
                yerr_=None
            else:
                yerr_=yerr[i]
            if label is None:
                label_ = 'Messwerte'
            else:
                if type(label) is str:
                    label_ = label
                else:
                    label_ = label[i]
            param, error = plotfit(x[i],y[i],f,savepath,slice_=slice_,yerr=yerr_, p0=p0, save = False, color=colors[i], label = label_, fitlabel = fitlabel[i])
            params = np.append(params, [param], axis = 0)
            errors = np.append(errors, [error], axis = 0)
    else:
        if yerr is None:
            plt.plot(x,y, color=color, linestyle='', marker='x', label =label)
        else:
            plt.errorbar(x,y,yerr=yerr, color=color, linestyle='', marker='x', label =label)
        params, covariance_matrix = curve_fit(f, x[slice_], y[slice_],p0=p0)
        errors = np.sqrt(np.diag(covariance_matrix))
        x_plot = np.linspace(np.min(x[slice_]), np.max(x[slice_]), 1000)
        if fitlabel is None:
            fitlabel=f.__name__
        plt.plot(x_plot, f(x_plot, *params), color=color, linestyle='-', label=fitlabel)
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

#
phi, U_min, U_max = np.genfromtxt('scripts/data/data_K.txt', unpack=True)
K = (U_max - U_min)/(U_max + U_min)

#plot datapoints
plt.plot(phi, K, 'rx', label = 'Messwerte')

#fit-funktion: K_0 ist korrektur wegen nicht perfekter Justage
def Fit(phi, K_0, delta):
    return K_0 * np.abs(np.sin((2 * phi * 2 * cst.pi/360 + delta * 2 * cst.pi/360)))

#fit datapoints
params_K, covariance_matrix = curve_fit(Fit, phi, K)
errors_K = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(np.min(phi), np.max(phi), 1000)
plt.plot(x_plot, Fit(x_plot, *params_K), 'b-', label='Fit')


plt.xlabel(r'$\Theta/°$')
plt.ylabel(r'$K/1$')
plt.legend(loc = 'best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_K.pdf')
plt.clf()

print("Kontrast")
print("Maximaler Kontrast: ", np.max(K))
print("Winkel des maximalen Kontrasts: ", phi[np.argmax(K)])
print("")

#Glas

run_Glas, M_glas = np.genfromtxt('scripts/data/data_counts.txt', unpack=True)

lam = 632.99 * 10**(-9)
T = 20.5 + 273.15
Theta_0 = 10/360 * 2 * cst.pi

def n_Glas_(M):
    return 1/(1- M * lam/(2 * T * Theta_0 **2))

n_Glas = n_Glas_(M_glas)

print("-------------------")
print("Brechungsindex von Glas:")
print("Run                    n")
print(np.array([run_Glas, n_Glas]).T)
print("-------------------")
print("Mittelwert:")
print("n = 1 + ", np.mean(n_Glas)-1, " \pm ", np.std(n_Glas))
print("")


#Gas
p, M_Gas_1, M_Gas_2, M_Gas_3 = np.genfromtxt('scripts/data/data_counts_pressure.txt', unpack=True)

L = 0.1

def n_Gas_(M):
    return M * lam/L + 1

def n_Gas_p(p, a, b): #quadriert
    return a * p + b

M_Gas = np.array([M_Gas_1, M_Gas_2, M_Gas_3])
n_Gas = n_Gas_(M_Gas)

# plt.xlabel(r'$p/\si{\milli\bar}$')
plt.ylabel(r'$n^2$')
params, errors = plotfit(np.array([p, p, p]), n_Gas**2, n_Gas_p, 'build/plot_n_Gas.pdf', label = np.array(["Run 1", "Run 2", "Run 3"]), fitlabel = np.array(["Fit 1", "Fit 2", "Fit 3"]))

a_1 = unp.uarray(params[0,0], errors[0,0])
b_1 = unp.uarray(params[0,1], errors[0,1])
a_2 = unp.uarray(params[1,0], errors[1,0])
b_2 = unp.uarray(params[1,1], errors[1,1])
a_3 = unp.uarray(params[2,0], errors[2,0])
b_3 = unp.uarray(params[2,1], errors[2,1])
T_normal = 15+273.15
print(T)
print(T/T_normal)
n_Normal_1 = unp.sqrt(n_Gas_p(1013, a_1, b_1) * T / T_normal)
n_Normal_2 = unp.sqrt(n_Gas_p(1013, a_2, b_2) * T / T_normal)
n_Normal_3 = unp.sqrt(n_Gas_p(1013, a_3, b_3) * T / T_normal)

print("-------------------")
print("Fit-parameter:")
print("a                         b")
print(np.array([params, " \pm ", errors]))
print("-------------------")
print("Brechungindex bei 1013mbar und 15°C")
print(noms(n_Normal_1), " \pm ", std(n_Normal_1))
print(noms(n_Normal_2), " \pm ", std(n_Normal_2))
print(noms(n_Normal_3), " \pm ", std(n_Normal_3))
