import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cst
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as std)

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

# def Fit(x,a,b):
#     return x*a +b
# x,y = np.genfromtxt('scripts/data.txt',unpack=True)
# plt.xlabel('X')
# plt.ylabel('Y')
# plotfit(x,y,Fit,'build/plot.pdf')

#T1
tau_T1, U_T1 = np.genfromtxt("scripts/data/T1.csv", delimiter = ",", unpack = True, skip_header = 1)

tau_T1 /= 1000
U_T1 /= 1000

plt.plot(tau_T1, U_T1, 'k-'#, linewidth = '.3'
                    , label = 'Volle Daten')

#choose datapoints
slice_ = slice(0,15)

#plot datapoints
plt.plot(tau_T1[slice_], U_T1[slice_], 'rx', label = 'Verwendete Werte')

#fit datapoints
def Fit(tau, M0, M1, T1):
    return M0 * np.exp(-tau/T1) + M1

params_T1, covariance_matrix = curve_fit(Fit, tau_T1[slice_], U_T1[slice_], p0=(-2, 1, 2))
errors_T1 = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(np.min(tau_T1[slice_]), np.max(tau_T1[slice_]), 1000)
plt.plot(x_plot, Fit(x_plot, *params_T1), 'b-', label='Fit')


plt.xlabel(r'$\tau/\si{\second}$')
plt.ylabel(r'$U/\si{\volt}$')

plt.xscale('log')
#plt.yscale('symlog')
plt.legend(loc = 'best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/T1.pdf')
plt.clf()


print("T_1-Fit:")
print("M_0 = ", params_T1[0], " \\pm ", errors_T1[0], '/V')
print("M_1 = ", params_T1[1], " \\pm ", errors_T1[1], '/V')
print("T_1 = ", params_T1[2], " \\pm ", errors_T1[2], '/s')
print()

#T2
t_T2, U_T2 = np.genfromtxt("scripts/data/T2_MG.csv", delimiter = ",", unpack = True, skip_header = 2)

plt.plot(t_T2, U_T2, 'k-', linewidth = '.3', label = 'Volle Daten')

#choose datapoints
slice_ = U_T2 > 100000
slice_[855] = True
slice_[896] = True
slice_[937] = True
slice_[947] = True
slice_[978] = True
slice_[988] = True
#slice_[998] = True #
#slice_[1009] = True #
#slice_[1019] = True
slice_[1029] = True
#slice_[1039] = True
#slice_[1060] = True
slice_[1070] = True
#slice_[1080] = True
#slice_[1101] = True
slice_[1111] = True
#slice_[1121] = True
slice_[1152] = True
#slice_[1162] = True
slice_[1193] = True
slice_[1203] = True
#slice_[1234] = True
slice_[1244] = True
slice_[1285] = True
slice_[1326] = True
slice_[1367] = True
slice_[1408] = True
slice_[1449] = True
slice_[1459] = True
slice_[1490] = True
slice_[1500] = True
slice_[1541] = True
slice_[1582] = True
slice_[1623] = True
slice_[1664] = True
slice_[1674] = True
slice_[1705] = True
slice_[1715] = True
slice_[1746] = True
slice_[1756] = True
slice_[1797] = True
slice_[1838] = True

#plot datapoints
plt.plot(t_T2[slice_], U_T2[slice_], 'rx', label = 'Verwendete Werte')

#fit datapoints
def Fit(t, M0, M1, T2):
    return M0 * np.exp(-t/T2) + M1

params_T2, covariance_matrix = curve_fit(Fit, t_T2[slice_], U_T2[slice_])
errors_T2 = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(np.min(t_T2[slice_]), np.max(t_T2[slice_]), 1000)
plt.plot(x_plot, Fit(x_plot, *params_T2), 'b-', label='Fit')


plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$U/\si{\volt}$')
plt.legend(loc = 'best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/T2.pdf')
plt.clf()

print("T_2-Fit:")
print("M_0 = ", params_T2[0], " \\pm ", errors_T2[0], '/V')
print("M_1 = ", params_T2[1], " \\pm ", errors_T2[1], '/V')
print("T_2 = ", params_T2[2], " \\pm ", errors_T2[2], '/s')
T_2 = params_T2[2]
T_2_err = errors_T2[2]

#Diffusion

#Laden der Daten aus der Datei "echo_gradient.csv"
#Die erste Spalte enthält die Zeiten in Sekunden, die zweite Spalte
#den Realteil und die dritte Spalte den Imaginärteil
t_diff, real, imag = np.genfromtxt("scripts/data/scope_20.csv", delimiter=",", skip_header=3, unpack= True)

#Suchen des Echo-Maximums und alle Daten davor abschneiden
start = np.argmax(real)
t_diff = t_diff[start:]
real = real[start:]
imag = imag[start:]

#Phasenkorrektur - der Imaginärteil bei t=0 muss = 0 sein
phase = np.arctan2(imag[0], real[0])

#Daten in komplexes Array mit Phasenkorrektur speichern
compsignal = (real*np.cos(phase)+imag*np.sin(phase))+(-real*np.sin(phase)+imag*np.cos(phase))*1j

#Offsetkorrektur, ziehe den Mittelwert der letzten 512 Punkte von allen Punkten ab
compsignal = compsignal - compsignal[-512:-1].mean()

#Der erste Punkt einer FFT muss halbiert werden
compsignal[0] = compsignal[0]/2.0

#Anwenden einer Fensterfunktion (siehe z. Bsp.
#https://de.wikipedia.org/wiki/Fensterfunktion )
#Hier wird eine Gaußfunktion mit sigma = 100 Hz verwendet
apodisation = 100.0*2*np.pi
compsignal = compsignal*np.exp(-1.0/2.0*((t_diff-t_diff[0])*apodisation)**2)

#Durchführen der Fourier-Transformation
fftdata = np.fft.fftshift(np.fft.fft(compsignal))

#Generieren der Frequenzachse
freqs = np.fft.fftshift(np.fft.fftfreq(len(compsignal), t_diff[1]-t_diff[0]))

#Speichern des Ergebnisses als txt
np.savetxt("echo_gradient_fft.txt", np.array([freqs, np.real(fftdata), np.imag(fftdata)]).transpose())

#Zoom in den interessanten Bereich
mask = freqs > -7200
mask[freqs>8000] = False
d_f = 15200


#Erstellen eines Plots
plt.plot(freqs[mask], np.real(fftdata[mask]))
plt.xlabel(r'$f/\si{\hertz}$')
plt.savefig("build/echo_gradient.pdf")
plt.clf()


#Magnetfeldgradient
gamma = 42.6 * 10**6# * 2 * cst.pi
d= 0.0044

#G = 2 * pi * d_f / (gamma * d)
G = d_f/(gamma * d)

print("")
print("Gradientenstärke")
print(G)

#Diffusionszeit
def Fit(tau, M0, M1, a):
    return M0 * np.exp(-2*tau/T_2) * np.exp(-2*tau**3/a) + M1

#D = 3 * a/(gamma**2 * G**2)


tau_D = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) * 10**(-3)
U_D = np.empty(29)

for i in range(0, 29):
    path = "scripts/data/scope_"
    path += str(i + 8)
    path += ".csv"
    t, U, imag = np.genfromtxt(path, delimiter = ",", skip_header = 3, unpack = True)
    mask = np.isfinite(U)
    U_D[i] = np.max(U[mask])


plt.plot(tau_D*1000, U_D, 'rx', label = 'Messwerte')

slice_ = slice(None,None)

#it took too long to actually figure out, so brute force it is. Conclusion: p0 = (1, 5, 0) works (even tho 0 causes true divide)
#condition = True
#while(condition):
#    p0_ = (np.random.randint(-5, 5+1), np.random.randint(-5, 5+1), np.random.randint(-5, 5+1))
#    params_D, covariance_matrix = curve_fit(Fit, tau_D[slice_], U_D[slice_], p0 = p0_)
#    errors_D = np.sqrt(np.diag(covariance_matrix))
#    if(np.all(np.isfinite(errors_D))):
#        if(np.all(np.abs(errors_D/params_D)<0.5)):
#            condition = False
#    print(p0_)


print("")
print("Ignorier die Fehlermeldung")
params_D, covariance_matrix = curve_fit(Fit, tau_D[slice_], U_D[slice_], p0 = (1, 5, 0))
errors_D = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(np.min(tau_D[slice_]), np.max(tau_D[slice_]), 1000)
plt.plot(x_plot*1000, Fit(x_plot, *params_D), 'b-', label='Fit')


plt.xlabel(r'$\tau/\si{\milli\second}$')
plt.ylabel(r'$U/\si{\volt}$')

plt.legend(loc = 'best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/tau_D.pdf')
plt.clf()

print("")
print("Tau_d-Fit:")
print("M_0 = ", params_D[0], " \\pm ", errors_D[0], '/V')
print("M_1 = ", params_D[1], " \\pm ", errors_D[1], '/V')
print("a = ",   params_D[2], " \\pm ", errors_D[2], '/s^3')

a = unp.uarray(params_D[2], errors_D[2])

#T_D = a/tau**2
#D = T_D * 3 / (gamma * G * tau)**2
D = 3 * 1/(a * (gamma * 2 * cst.pi * G)**2)
print("")
print("Dispersion")
print("D = ", noms(D), "\\pm", std(D))

#Molekülradius
T = 20.6 + 273.15
ŋ = 10**(-3) #Quelle Wikipedia \cite{Viskosität_Wasser}
r = cst.k * T/(6 * cst.pi * ŋ * D)
print("")
print("Molekülradius:")
print("r = ", noms(r), "\\pm", std(r))
