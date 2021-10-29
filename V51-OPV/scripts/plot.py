import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
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

#a)

f_a1, U_a_a1, phi_a1 = np.genfromtxt('scripts/data_a1.txt', unpack=True)
f_a2, U_a_a2, phi_a2 = np.genfromtxt('scripts/data_a2.txt', unpack=True)
f_a3, U_a_a3, phi_a3 = np.genfromtxt('scripts/data_a3.txt', unpack=True)

f_a = np.array([f_a1, f_a2, f_a3])
U_a_a = np.array([U_a_a1, U_a_a2, U_a_a3])

U_e_a = 0.05

plt.xlabel(r'$f$/[Hz]')
plt.xscale('log')
plt.ylabel(r'$\left|\frac{\text{max}\left(U_a\right)}{\text{max}\left(U_e\right)}\right|$')
plt.yscale('log')


plt.plot(f_a1, U_a_a1/U_e_a, color='blue',  linestyle='', marker='x', label=r'$V_{Theorie}=100$')
plt.plot(f_a2, U_a_a2/U_e_a, color='black', linestyle='', marker='x', label=r'$V_{Theorie}=15$')
plt.plot(f_a3, U_a_a3/U_e_a, color='green', linestyle='', marker='x', label=r'$V_{Theorie}=3.3$')

def Fit(x, a, b):
    return np.exp(np.log(x)*a + b)

params_a = np.empty((3, 2))
errors_a = np.empty((3, 2))
colors_ = ['blue', 'black', 'green']
slice_ = [slice(7,-2), slice(7,-2), slice(-2,None)]

for i in range(0, 2):
    params_a[i], covariance_matrix = curve_fit(Fit, f_a[i][slice_[i]], U_a_a[i][slice_[i]]/U_e_a)
    errors_a[i] = np.sqrt(np.diag(covariance_matrix))
    x_plot = np.linspace(np.min(f_a[i][slice_[i]]), np.max(f_a[i][slice_[i]]), 1000)
    plt.plot(x_plot, Fit(x_plot, *params_a[i]), color=colors_[i], linestyle='-')

plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_U_f_a.pdf')
plt.clf()

V_a1 = np.mean(U_a_a1[0:5]/U_e_a)
V_a2 = np.mean(U_a_a2[0:4]/U_e_a)
V_a3 = np.mean(U_a_a3[0:10]/U_e_a)

a_a1 = unp.uarray(params_a[0,0], errors_a[0,0])
b_a1 = unp.uarray(params_a[0,1], errors_a[0,1])
a_a2 = unp.uarray(params_a[1,0], errors_a[1,0])
b_a2 = unp.uarray(params_a[1,1], errors_a[1,1])

f_Grenz_a1 = unp.exp((unp.log(V_a1/np.sqrt(2))-b_a1)/a_a1)
f_Grenz_a2 = unp.exp((unp.log(V_a2/np.sqrt(2))-b_a2)/a_a2)

print("Teil a)")
print("Verstärkung 1 (V_Theorie=100): ", V_a1)
print("Verstärkung 2 (V_Theorie=15): ", V_a2)
print("Verstärkung 3 (V_Theorie=3.3): ", V_a3)
print("--------------------------------------")
print("Ausgleichsgerade 1: a = ", a_a1, "; b = ", b_a1)
print("Ausgleichsgerade 2: a = ", a_a2, "; b = ", b_a2)
print("--------------------------------------")
print("Grenzfrequenz 1: f = ", f_Grenz_a1)
print("Grenzfrequenz 2: f = ", f_Grenz_a1)
print("--------------------------------------")
print("VBP 1: ", V_a1*f_Grenz_a1)
print("VBP 2: ", V_a2*f_Grenz_a2)


#b)

plt.xlabel(r'$f$/[Hz]')
plt.xscale('log')
plt.ylabel(r'$\phi/°$')

plt.plot(f_a1[phi_a1>0], phi_a1[phi_a1>0], color='blue',  linestyle='', marker='x', label=r'$V_{Theorie}=100$')
plt.plot(f_a1[phi_a2>0], phi_a2[phi_a2>0], color='black', linestyle='', marker='x', label=r'$V_{Theorie}=15$')
plt.plot(f_a1[phi_a3>0], phi_a3[phi_a3>0], color='green', linestyle='', marker='x', label=r'$V_{Theorie}=3.3$')

plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_phi_f_a.pdf')
plt.clf()


#c)

f_c, U_a_c1, U_a_c2 = np.genfromtxt('scripts/data_c.txt', unpack=True)

plt.xlabel(r'$f$/[Hz]')
plt.xscale('log')
plt.ylabel(r'$\left|\frac{\text{max}\left(U_a\right)}{\text{max}\left(U_e\right)}\right|$')
plt.yscale('log')


plt.plot(f_c, U_a_c1/U_e_a, color='blue',  linestyle='', marker='x', label=r'Messreihe 1')
plt.plot(f_c, U_a_c2/U_e_a, color='black', linestyle='', marker='x', label=r'Messreihe 2')

plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_U_f_c.pdf')
plt.clf()

#d)

def Fit(x, a, b):
    return np.exp(a*np.log(x)+b)

f_d, U_a_d = np.genfromtxt('scripts/data_d.txt', unpack=True)

plt.xlabel(r'$f$/[Hz]')
plt.xscale('log')
plt.ylabel(r'$\left|\frac{\text{max}\left(U_a\right)}{\text{max}\left(U_e\right)}\right|$')
plt.yscale('log')


plt.plot(f_d, U_a_d/U_e_a, color='blue',  linestyle='', marker='x', label=r'Messwerte')

slice_ = slice(7,-1)
params_d, covariance_matrix = curve_fit(Fit, f_d[slice_], U_a_d[slice_]/U_e_a)
errors_d = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(np.min(f_d[slice_]), np.max(f_d[slice_]), 1000)
plt.plot(x_plot, Fit(x_plot, *params_d), color='blue', linestyle='-', label='Fit')

plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_U_f_d.pdf')
plt.clf()

a_d = unp.uarray(params_d[0], errors_d[0])
b_d = unp.uarray(params_d[1], errors_d[1])

print("")
print("--------------------------------------")
print("")
print("Teil d)")
print("Ausgleichsgerade: a = ", a_d, "; b = ", b_d)


#e)

R_1 = 100
R_2 = 220*10**3
U_versorgung = 15

U_schwell = U_versorgung*R_1/R_2

print("")
print("--------------------------------------")
print("")
print("Teil e)")
print("Theorethische Schwellspannung: ", U_schwell, "V")