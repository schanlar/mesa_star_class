import numpy as np
import matplotlib.pyplot as plt
from mesa_star_class import MESA_STAR
import os
import astropy.units as u
import astropy.constants as c
from functools import wraps

mesa_dir = '/Users/SavvasGCh/mesa-r10398'
plot_results_dir = '/Users/SavvasGCh/Desktop'


def plot_decorator(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        print('Plotting in progress...')
        func(*args, **kwargs)
        print('All done! \n')

    return wrapper

def prepare_canvas():
    '''
    The basic canvas for plots
    '''

    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['axes.linewidth'] = 2 #3

    fontsize = 15 #20
    ax = plt.gca()
    ax.tick_params(direction='in',length=5)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

def capture_density(t,rho_0,Q,t_comp,ft):
    '''
    Density for electron captures
    '''

    rho = rho_0/(1 + (3*c.k_B*t/Q)* np.log(2*np.log(2)*(c.k_B*t/(c.m_e*c.c**2))**5 * (Q/(c.k_B*t))**2 * (t_comp/ft)))
    return rho

def burning_regions(mesa_dir = mesa_dir,
                xlim=None,
                ylim=None,
                ecap_density_corrections=True,
                t_comp=1e4*u.yr):

    '''
    Define various burning and other relative regions according
    to the data stored in $MESA_DIR
    '''


    # hydrogen_burning_line = os.path.join(mesa_dir,'data/star_data/plot_info/hydrogen_burn.data')
    helium_burning_line = os.path.join(mesa_dir,'data/star_data/plot_info/helium_burn.data')
    carbon_burning_line = os.path.join(mesa_dir,'data/star_data/plot_info/carbon_burn.data')
    oxygen_burning_line = os.path.join(mesa_dir,'data/star_data/plot_info/oxygen_burn.data')
    electron_degeneracy_line = os.path.join(mesa_dir,'data/star_data/plot_info/psi4.data')



    # hburn = np.genfromtxt(hydrogen_burning_line)
    heburn = np.genfromtxt(helium_burning_line)
    cburn = np.genfromtxt(carbon_burning_line)
    oburn = np.genfromtxt(oxygen_burning_line)
    electron = np.genfromtxt(electron_degeneracy_line)


    # Radiation pressure line
    logrho = np.arange(-9.0,10.0,0.1)
    logt = np.log10(3.2e7) + (logrho - np.log10(0.7))/3.0


    plt.plot(heburn[:,0],heburn[:,1],ls=':',color='black')
    plt.text(5.1, 7.95, 'He burn', fontsize=22,
            rotation=0, rotation_mode='anchor')


    plt.plot(cburn[:,0],cburn[:,1],ls=':',color='black')
    plt.text(5.1, 8.67, 'C burn', fontsize=22,
            rotation=0, rotation_mode='anchor')


    plt.plot(oburn[:,0],oburn[:,1],ls=':',color='black')
    plt.text(5.1, 9.05, 'O burn', fontsize=22,
            rotation=0, rotation_mode='anchor')

    plt.plot(electron[:,0],electron[:,1],ls='--',color='black')

    plt.plot(logrho,logt,ls='--',color='black')

    plt.text(7.0, 9.5, r'$\epsilon_{\rm F}/k T \simeq 4$', fontsize=22, rotation=0, rotation_mode='anchor')

    plt.text(5.12, 9.5, r'$P_{\rm rad}\simeq P_{\rm gas}$', fontsize=22, rotation=0, rotation_mode='anchor')


    # Weak reaction lines
    plt.text(9.05, 7.52, r'$^{25}{\rm Mg}\leftrightarrow ^{25}{\rm Na}$', fontsize=15, rotation=90,verticalalignment='bottom')
    plt.text(9.25, 7.52, r'$^{23}{\rm Na} \leftrightarrow ^{23}{\rm Ne}$', fontsize=15, rotation=90,verticalalignment='bottom')
    plt.text(9.65, 7.52, r'$^{24}{\rm Mg}\rightarrow ^{24}{\rm Na}$', fontsize=15, rotation=90,verticalalignment='bottom')
    plt.text(9.75, 7.52, r'$^{24}{\rm Na}\rightarrow ^{24}{\rm Ne}$', fontsize=15, rotation=90,verticalalignment='bottom')
    plt.text(9.85, 7.52, r'$^{25}{\rm Na}\leftrightarrow ^{25}{\rm Ne}$', fontsize=15, rotation=90,verticalalignment='bottom')
    plt.text(10.00, 7.52, r'$^{20}{\rm Ne}\rightarrow ^{20}{\rm F}\rightarrow  ^{20}{\rm O}$', fontsize=15, rotation=90,verticalalignment='bottom')


    if ecap_density_corrections:
        t = np.arange(7.5,11,0.1)
        t = 10**t * u.K
        rho_ce = capture_density(t,10**9.96,7.025*u.MeV,t_comp,10**9.801*u.s)
        plt.plot(np.log10(rho_ce),np.log10(t.value),color='red',ls='--')
    else:
        plt.axvline(x=9.96,color='red',ls='-')

    plt.text(10.0, 8.3, r'$e^{-}$cSN', fontsize=15, rotation=90,color='red',verticalalignment='bottom')


# Plotters

@plot_decorator
def plotRhoT(star,
    xlim=None,
    ylim=None,
    saveFigure=False,
    figureName='Rhoc_vs_Tc.pdf',
    plot_output_dir=plot_results_dir):

    '''
    It takes a MESA_STAR object as a mandatory argument,
    and it plots the (log) central density vs (log) central temperature
    diagram.
    '''

    prepare_canvas()
    burning_regions()

    h = star.getHistory()

    labels = ['LM;WNO', 'LM;WO1', 'LM;WO2', 'IM;WNO', 'IM;WO1', 'IM;WO2', 'SM;WNO', 'SM;WO1', 'SM;WO2']

    tag1 = str(round(float(star.getMass()), 1)) + r'M$_{\odot}$'

    if star.getMetallicity() == '0.0001' and star.getOvershoot() == '0.0000':
        tag2 = labels[0]
    elif star.getMetallicity() == '0.0001' and star.getOvershoot() == '0.0140':
        tag2 = labels[1]
    elif star.getMetallicity() == '0.0001' and star.getOvershoot() == '0.0160':
        tag2 = labels[2]
    elif star.getMetallicity() == '0.0010' and star.getOvershoot() == '0.0000':
        tag2 = labels[3]
    elif star.getMetallicity() == '0.0010' and star.getOvershoot() == '0.0140':
        tag2 = labels[4]
    elif star.getMetallicity() == '0.0010' and star.getOvershoot() == '0.0160':
        tag2 = labels[5]
    elif star.getMetallicity() == '0.0200' and star.getOvershoot() == '0.0000':
        tag2 = labels[6]
    elif star.getMetallicity() == '0.0200' and star.getOvershoot() == '0.0140':
        tag2 = labels[7]
    elif star.getMetallicity() == '0.0200' and star.getOvershoot() == '0.0160':
        tag2 = labels[8]

    plt.plot(h.data('log_center_Rho'), h.data('log_center_T'), label = tag1 + ', ' + tag2)

    legend = plt.legend(loc = 'upper left', prop = {'size':12}, bbox_to_anchor=(1, 1), shadow = False)

    #frame & labels
    xlabel = r'$\log (\rho_{\rm c} / {\rm gr}\,{\rm cm}^{-3})$'
    ylabel = r'$\log (T_{\rm c} / {\rm K})$'
    plt.xlabel(xlabel, fontsize=23)
    plt.ylabel(ylabel, fontsize=23)

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([5,10.5])
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim([7.5,10.0])


    if saveFigure:
        plt.savefig(os.path.join(plot_output_dir, figureName), bbox_inches = 'tight', dpi =300)
        plt.clf()
    else:
        plt.show()





def main():

    #print(plotRhoT.__name__)

    print(
        'Show more info for:' + '\n' + \
        '1: The prepare_canvas function' + '\n' + \
        '2: The capture_density function' + '\n' + \
        '3: The burning_regions function' + '\n' + \
        '4: The plotRhoT function' + '\n'
    )

    answer = input('Press the corresponding number: ')

    if answer == '1':
        help(prepare_canvas)
    elif answer == '2':
        help(capture_density)
    elif answer == '3':
        help(burning_regions)
    elif answer == '4':
        help(plotRhoT)
    else:
        print('Invalid input!')

if __name__ == '__main__':
    main()
