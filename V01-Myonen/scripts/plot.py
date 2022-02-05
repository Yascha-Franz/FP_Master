import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import types

def plotfit(x,y,f,savepath,slice_=slice(0,None),yerr=None, p0=None, save=True, color='b', label= None, fitlabel = None):
    colors = ['b', 'g', 'r', 'y']
    if label is None:
        label = 'Messwerte'
    if (np.size(x[0])>1):
        if yerr==None:
            yerr_=None
        else:
            yerr_=yerr[0]
        if type(label) is str:
            label_ = label
        else:
            label_ = label[0]
        if type(f) is types.FunctionType:
            f_ = f
        else:
            f_ = f[0]
        param, error = plotfit(x[0],y[0],f_,savepath,slice_=slice_,yerr=yerr_, p0=p0, save = False, color=colors[0], label = label_, fitlabel = fitlabel[0])
        params = np.array([param])
        errors = np.array([error])
        for i in range(1,np.shape(x)[0]):
            if yerr==None:
                yerr_=None
            else:
                yerr_=yerr[i]
            if type(label) is str:
                label_ = label
            else:
                label_ = label[i]
            if type(f) is types.FunctionType:
                f_ = f
            else:
                f_ = f[i]
            param, error = plotfit(x[i],y[i],f_,savepath,slice_=slice_,yerr=yerr_, p0=p0, save = False, color=colors[i], label = label_, fitlabel = fitlabel[i])
            params = np.append(params, np.array([param]), axis = 0)
            errors = np.append(errors, np.array([error]), axis = 0)
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

#Kalibrierung Koinzidenz

delta_t_10_, counts_left_10, counts_right_10 = np.genfromtxt('scripts/verzoegerung_10ns.txt', unpack = True)

delta_t_20_, counts_left_20, counts_right_20 = np.genfromtxt('scripts/verzoegerung_20ns.txt', unpack = True)


delta_t_10 = np.append(- np.flip(delta_t_10_[1:]), delta_t_10_)
counts_10 = np.append(np.flip(counts_left_10[1:]), counts_right_10)

delta_t_20 = np.append(- np.flip(delta_t_20_[1:]), delta_t_20_)
counts_20 = np.append(np.flip(counts_left_20[1:]), counts_right_20)

cut_10_1 = 4
cut_10_2 = 12

cut_20_1 = 5
cut_20_2 = 23

def Fit_Flanke(x, a, b):
    return x * a + b

def Fit_Plateau(x, a, b): #Aus faulheit so miteinbezogen. Erlaubt mir meine plotfit-funktion zu verwenden. Egentlich nur np.mean(), gibt aber gleiche Ergebnisse
    return 0 * x * a + b

plt.xlabel(r'$\Delta t/\si{\nano\second}$')
plt.ylabel('counts')
params_10, errors_10 =  plotfit([delta_t_10[:cut_10_1], delta_t_10[cut_10_1:cut_10_2], delta_t_10[cut_10_2:]], [counts_10[:cut_10_1], counts_10[cut_10_1:cut_10_2], counts_10[cut_10_2:]], [Fit_Flanke, Fit_Plateau, Fit_Flanke], 'build/koinzidenz_10ns.pdf', fitlabel = ['Fit linke Flanke', 'Fit Plateau', 'Fit rechte Flanke'])

plt.xlabel(r'$\Delta t/\si{\nano\second}$')
plt.ylabel('counts')
params_20, errors_20 =  plotfit([delta_t_20[:cut_20_1], delta_t_20[cut_20_1:cut_20_2], delta_t_20[cut_20_2:]], [counts_20[:cut_20_1], counts_20[cut_20_1:cut_20_2], counts_20[cut_20_2:]], [Fit_Flanke, Fit_Plateau, Fit_Flanke], 'build/koinzidenz_20ns.pdf', fitlabel = ['Fit linke Flanke', 'Fit Plateau', 'Fit rechte Flanke'])


print("Koinzidenz")
print()
print("Parameter für 10ns")
print("Linke Flanke:")
print("a = ", params_10[0,0], " \pm ", errors_10[0,0])
print("b = ", params_10[0,1], " \pm ", errors_10[0,1])
print("Plateau:")
print(np.mean(counts_10[cut_10_1:cut_10_2]), " \pm ", np.std(counts_10[cut_10_1:cut_10_2]))
print("Rechte Flanke:")
print("a = ", params_10[2,0], " \pm ", errors_10[2,0])
print("b = ", params_10[2,1], " \pm ", errors_10[2,1])
print()
print("Parameter für 20ns")
print("Linke Flanke:")
print("a = ", params_20[0,0], " \pm ", errors_20[0,0])
print("b = ", params_20[0,1], " \pm ", errors_20[0,1])
print("Plateau:")
print(np.mean(counts_20[cut_20_1:cut_20_2]), " \pm ", np.std(counts_20[cut_20_1:cut_20_2]))
print("Rechte Flanke:")
print("a = ", params_20[2,0], " \pm ", errors_20[2,0])
print("b = ", params_20[2,1], " \pm ", errors_20[2,1])
print()





#Kalibrierung Zerfallszeit-Channel

t, channel = np.genfromtxt('scripts/pulsabstaende.txt', unpack = True)

def Fit_Zeit(channel, a, b):   #t(channel)
    return channel * a + b

#channel(t) = (t - b)/a


plt.ylabel(r'$t/\si{\micro\second}$')
plt.xlabel('Channel')
params_t_C, errors_t_C = plotfit(channel, t, Fit_Zeit, 'build/Zeit_Channel_Kalibration.pdf')

a = params_t_C[0]
b = params_t_C[1]

print("---------------------")
print()
print("Zeit-Channel-Kalibration")
print()
print("a = ", a, " \pm ", errors_t_C[0])
print("b = ", b, " \pm ", errors_t_C[1])

#Doppelmyonen-Untergrund
T_s = 10 #µs Suchzeit
T_activation = 0.03 #µs Verzögerungszeit des Startsignals

Gesamtspektrum = np.genfromtxt('scripts/31-01-2022.Spe', unpack=True, skip_header=12, skip_footer=14)

total_starts = 9530546
total_stops = 10895
total_channels = np.size(Gesamtspektrum)
total_time = 266036
event_rate = total_starts/total_time / 10**6
x = np.arange(start = 0, stop = total_channels, step = 1)
t_decay_ = Fit_Zeit(x, *params_t_C)


#nicht messbare Einträge entfernen
slice_ = t_decay_ < T_s #Einträge über Suchzeit
slice_[0:4] = False     #leere Einträge
t_decay = t_decay_[slice_]
Gesamtspektrum = Gesamtspektrum[slice_]


from scipy.stats import poisson


def underground(t, rate, total_events): #poissonverteilt mit lambda = rate * time
    P = poisson.pmf(k=1, mu=t * rate)   #Die wahrscheinlichkeit innerhalb eines gewissen Zeitraums t zwei muon-durchläufe zu messen
    if np.size(t)>1:
        P_ = np.copy(P)
        for i in range(1,np.size(t)):
            P[i] = P_[i]-P_[i-1]        #Wahrscheinlichkeit in dem spezifischen Bin den Durchlauf zu messen
    return P * total_events             #P * events = volle Anzahl an Doppel-Muon-durchläufen

Untergrund = underground(t_decay_, event_rate, total_starts)[slice_]

Counts_Zusammen = []
Channel_Zusammen = []
Untergrund_Zusammen = []
for i in range(1, np.size(Gesamtspektrum), 2):
    Counts_Zusammen.append(Gesamtspektrum[i] + Gesamtspektrum[i-1])
    Untergrund_Zusammen.append(Untergrund[i] + Untergrund[i-1])
    Channel_Zusammen.append(i + .5)
Counts_Zusammen = np.array(Counts_Zusammen)
Untergrund_Zusammen = np.array(Untergrund_Zusammen)
Channel_Zusammen = np.array(Channel_Zusammen)
t_Zusammen = Fit_Zeit(Channel_Zusammen, *params_t_C)


plt.xlabel(r'$t_{decay}/\si{\micro\second}$')
plt.ylabel('Counts')
#plt.step(t_decay, Untergrund_Zusammen, label='Untergrund')
plt.step(t_decay, Untergrund, label='Untergrund')
plt.vlines(1/event_rate, ymin=0, ymax=0.2)

plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Myonenuntergrund.pdf')
plt.clf()

def Fit_Zerfall(t, a, tau, c):
    return a * np.exp(- t/tau) + c


def plot_fein():
    plt.errorbar(t_decay, Gesamtspektrum - Untergrund, yerr = np.sqrt(Gesamtspektrum), color='b', linestyle='', marker='x', elinewidth = .3, markersize = 2, mew = .5, label='Gesamtspektrum')
    params_Zerfall, covariance_matrix = curve_fit(Fit_Zerfall, t_decay, Gesamtspektrum - Untergrund)
    errors_Zerfall = np.sqrt(np.diag(covariance_matrix))
    x_plot = np.linspace(np.min(t_decay), np.max(t_decay), 1000)
    plt.plot(x_plot, Fit_Zerfall(x_plot, *params_Zerfall), color='k', linestyle='-', label='Fit')
    return params_Zerfall, errors_Zerfall

def plot_grob():
    plt.errorbar(t_Zusammen, Counts_Zusammen - Untergrund_Zusammen, yerr = np.sqrt(Counts_Zusammen), color='b', linestyle='', marker='x', elinewidth = .3, markersize = 2, mew = .5, label='Gesamtspektrum')
    params_Zerfall, covariance_matrix = curve_fit(Fit_Zerfall, t_Zusammen, Counts_Zusammen - Untergrund_Zusammen)
    errors_Zerfall = np.sqrt(np.diag(covariance_matrix))
    x_plot = np.linspace(np.min(t_Zusammen), np.max(t_Zusammen), 1000)
    plt.plot(x_plot, Fit_Zerfall(x_plot, *params_Zerfall), color='k', linestyle='-', label='Fit')
    return params_Zerfall, errors_Zerfall

#params_Zerfall, errors_Zerfall = plot_grob()
params_Zerfall, errors_Zerfall = plot_fein()

plt.xlabel(r'$t_{decay}/\si{\micro\second}$')
plt.ylabel('Counts')


plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Gesamtspektrum.pdf')
plt.clf()

print("------------------------")
print()
print("Gesamt-zerfalls-spektrum")
print()
print("a = ",   params_Zerfall[0], " \pm ", errors_Zerfall[0])
print("tau = ", params_Zerfall[1], " \pm ", errors_Zerfall[1])
print("c = ",   params_Zerfall[2], " \pm ", errors_Zerfall[2])