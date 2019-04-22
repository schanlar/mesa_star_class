'''
@author Savvas Chanlaridis
@version v.16.04.19
@description https://github.com/schanlar/mesa_star_class
'''

import numpy as np
import mesa_reader as mr
import os
import astropy.units as u
import astropy.constants as c
from functools import wraps
import matplotlib.pyplot as plt





class MESA_STAR(object):


    # Paths to MESA stellar evolution code + output for plots

    mesa_dir = '/vol/aibn1107/data2/schanlar/mesa-r10398'
    plot_results_dir = '/users/schanlar/Desktop'



    # Main Constructor

    def __init__(self,
        mass: str,
        metallicity: str,
        overshooting: str,
        history_name = 'history',
        profile_number = 'final',
        **kwargs):

        '''
        The argument "profile_number" accepts either the number of a profile (e.g. 28, for profile28.data),
        or the word "final" as a default value which corresponds to the final profile (final_profile.data).

        The absolute paths for the history file, and a given profile can be set using the kwargs "history_path",
        and "profile_path" respectively.
        '''

        self.initial_mass = mass
        self.initial_metallicity = metallicity
        self.initial_overshooting = overshooting
        self.history_path = kwargs.get('history_path')
        self.profile_path = kwargs.get('profile_path')
        self.history_name = f'{history_name}.data'

        if not profile_number == 'final':
            self.profile_name = f'profile{profile_number}.data'
        else:
            self.profile_name = 'final_profile.data'

    # Output

    def __str__(self):
        return 'MESA_STAR[' + '\n' + \
            f'> Initial mass: {self.initial_mass} Msol' + '\n' + \
            f'> Initial metallicity: {self.initial_metallicity}'  + '\n' + \
            f'> Overshooting factor: {self.initial_overshooting}' + '\n' + \
            ']'


    # CLASS METHODS
    # --------------------------------------------------------------------------

    # Second Constructor

    @classmethod # This decorator accounts for function overloading as in C++
    def from_string(cls,
        input_as_string: str,
        history_name = 'history',
        profile_number = 'final',
        **kwargs):

        '''
        This constructor builds a MESA_STAR object when the user passes the
        info for mass, metallicity, and overshooting as a string with format
        "mass_metallicity_overshooting".

        The function parses the string and continues by calling the class.
        All other relative variables (e.g. history_name, profile_path etc)
        can/should be inserted separately.
        '''

        mass, metallicity, overshooting = map(str, input_as_string.split('_'))

        history_path = kwargs.get('history_path')
        profile_path = kwargs.get('profile_path')

        star = cls(mass,
            metallicity,
            overshooting,
            history_name,
            profile_number,
            **kwargs)

        return star





    # ACCESER METHODS (GETTERS)
    # --------------------------------------------------------------------------

    def getMass(self):
        '''
        Returns the initial mass of the star (type: str)
        '''
        return self.initial_mass

    def getMetallicity(self):
        '''
        Returns the initial metallicity of the star (type: str)
        '''
        return self.initial_metallicity

    def getOvershoot(self):
        '''
        Returns the overshooting factor of the star (type: str)
        '''
        return self.initial_overshooting

    def getHistory(self):
        '''
        This method exploits the mesa_reader module in order to load a MESA history file
        from a user-specified directory.

        It returns a <class 'mesa_reader.MesaData'> object.
        '''
        h = mr.MesaData(os.path.join(self.history_path, self.history_name))
        return h

    def getProfile(self):
        '''
        This method exploits the mesa_reader module in order to load a MESA profile file
        from a user-specified directory.

        It returns a <class 'mesa_reader.MesaData'> object.
        '''
        p = mr.MesaData(os.path.join(self.profile_path, self.profile_name))
        return p

    def getHistoryName(self):
        '''
        Returns the name of the MESA history file (type: str)
        '''
        return self.history_name

    def getProfileName(self):
        '''
        Returns the name of the MESA profile file (type: str)
        '''
        return self.profile_name

    def getHistoryPath(self):
        '''
        Returns the absolute path for the MESA history file (type: str)
        '''
        return self.history_path

    def getProfilePath(self):
        '''
        Returns the absolute path for the MESA profile file (type: str)
        '''
        return self.profile_path

    def getName(self):
        '''
        Returns the name of the star as a single string.
        Here as name we consider a string with the following
        format: mass_metallicity_overshooting
        '''
        a = [self.getMass(), self.getMetallicity(), self.getOvershoot()]
        self.name = '_'.join(a)
        return self.name

    def getCoreMass(self):
        '''
        Returns
            a float with the value of the initial star mass.
            a float with the value of the initial metallicity.
            a float with the value of the overshooting factor.
            a float with the value of the carbon core mass.
            a float with the value of the envelope mass.

        If MESA cannot distinguish between the carbon core mass,
        and the final mass of the star, then we define the outer
        boundary of the carbon core as the location where the
        pressure drops to 80% of its maximum value.

        If you're satisfied with this estimation, the function
        will return this value along with the initial mass,
        initial metallicity, overshooting factor, and envelope
        mass.

        It is also possible to set a new value for the core mass,
        ignoring the original approximation. In this case, a plot
        will appear every time you enter a new value for the core
        mass, in order to help visualize the mass cut.
        '''

        try:
            h = self.getHistory()
        except:
            raise SystemExit('Failed when trying to load history file!')


        if h.data('star_mass')[-1] != h.data('c_core_mass')[-1]:

            initial_mass = round(float(self.getMass()),1)
            initial_metallicity = float(self.getMetallicity())
            overshooting_factor = float(self.getOvershoot())

            final_core_mass = round(h.data('c_core_mass')[-1], 3)
            final_envelope_mass = round(h.data('star_mass')[-1] - h.data('c_core_mass')[-1], 3)

            return initial_mass, initial_metallicity, overshooting_factor, final_core_mass, final_envelope_mass

        elif h.data('star_mass')[-1] == h.data('c_core_mass')[-1]:

            try:
                p = self.getProfile()

                initial_mass = round(float(self.getMass()),1)
                initial_metallicity = float(self.getMetallicity())
                overshooting_factor = float(self.getOvershoot())

                mask = 0.80 * max(p.data('logP'))
                logP = p.data('logP')
                core_boundary = p.data('mass')[np.where(logP < mask)][-1]

                print(f'Final core mass estimate: {round(core_boundary,3)}')

                plt.figure(figsize = (13,9))
                plt.xlabel(r'Mass coordinate [M$_{\odot}$]')
                plt.ylabel(r'$\log(P)$ [Ba]')

                plt.plot(p.data('mass'), p.data('logP'))
                plt.axvline(core_boundary, c = 'r', linestyle = '--')
                plt.show()

                answer = input('Accept this estimation? (y/n) ')

                if answer == 'y' or answer == 'Y':
                    final_core_mass = round(core_boundary, 3)
                    final_envelope_mass = round(h.data('star_mass')[-1] - final_core_mass, 3)

                    return initial_mass, initial_metallicity, overshooting_factor, final_core_mass, final_envelope_mass

                else:

                    answer = input('Do you want to set a value for the core mass? (y/n) ')

                    if answer == 'y' or answer == 'Y':
                        tryAgain = True

                        while tryAgain:
                            core_boundary = float(input('set value: '))

                            plt.figure(figsize = (13,9))
                            plt.xlabel(r'Mass coordinate [M$_{\odot}$]')
                            plt.ylabel(r'$\log(P)$ [Ba]')

                            plt.plot(p.data('mass'), p.data('logP'))
                            plt.axvline(core_boundary, c = 'r', linestyle = '--')
                            plt.show()

                            answer = input('Try again? (y/n) ')

                            if answer == 'n' or answer == 'N':
                                tryAgain = False
                            else:
                                tryAgain = True

                        final_core_mass = round(core_boundary, 3)
                        final_envelope_mass = round(h.data('star_mass')[-1] - final_core_mass, 3)

                        return initial_mass, initial_metallicity, overshooting_factor, final_core_mass, final_envelope_mass

                    else:

                        final_core_mass = float('nan')
                        final_envelope_mass = float('nan')

                        return initial_mass, initial_metallicity, overshooting_factor, final_core_mass, final_envelope_mass


            except:
                raise SystemExit('WARNING: Failed to load profile! Make sure "logP" is defined in profile_columns.list')







    # MUTATOR METHODS (SETTERS)
    # --------------------------------------------------------------------------

    def setMass(self, new_mass: str):
        '''
        Set a new value for the initial mass (type: str)
        '''
        self.initial_mass = new_mass

    def setMetallicity(self, new_metallicity: str):
        '''
        Set a new value for the initial metallicity (type: str)
        '''
        self.initial_metallicity = new_metallicity

    def setOvershoot(self, new_overshoot: str):
        '''
        Set a new value for the overshooting factor (type: str)
        '''
        self.initial_overshooting = new_overshoot

    def setHistoryName(self, new_name: str):
        '''
        Set a new value for the name of the history file (type: str)
        '''
        self.history_name = new_name

    def setProfileName(self, new_name: str):
        '''
        Set a new value for the name of the profile file (type: str)
        '''
        self.profile_name = new_name

    def setHistoryPath(self, new_path: str):
        '''
        Set a new value for the absolute path of the history file (type: str)
        '''
        self.history_path = new_path

    def setProfilePath(self, new_path: str):
        '''
        Set a new value for the absolute path of the profile file (type: str)
        '''
        self.profile_path = new_path






    # PLOTTERS
    # --------------------------------------------------------------------------

    def __plot_decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            print('Plotting in progress...')
            func(*args, **kwargs)
            print('All done! \n')

        return wrapper



    def _prepare_canvas(self):
        '''
        The basic canvas for plots
        '''

        # FIXME: Change canvas 

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



    def _capture_density(self,t,rho_0,Q,t_comp,ft):
        '''
        Density for electron captures
        '''

        rho = rho_0/(1 + (3*c.k_B*t/Q)* np.log(2*np.log(2)*(c.k_B*t/(c.m_e*c.c**2))**5 * (Q/(c.k_B*t))**2 * (t_comp/ft)))
        return rho



    def _burning_regions(self,
                    mesa_dir = mesa_dir,
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
            rho_ce = self._capture_density(t,10**9.96,7.025*u.MeV,t_comp,10**9.801*u.s)
            plt.plot(np.log10(rho_ce),np.log10(t.value),color='red',ls='--')
        else:
            plt.axvline(x=9.96,color='red',ls='-')

        plt.text(10.0, 8.3, r'$e^{-}$cSN', fontsize=15, rotation=90,color='red',verticalalignment='bottom')



    @__plot_decorator
    def plotRhoT(self,
        xlim=None,
        ylim=None,
        saveFigure=False,
        figureName='Rhoc_vs_Tc.pdf',
        plot_output_dir = plot_results_dir):

        '''
        It plots the (log) central density vs (log) central temperature
        diagram of a MESA_STAR object.
        '''

        self._prepare_canvas()
        self._burning_regions()

        h = self.getHistory()

        labels = ['LM;WNO', 'LM;WO1', 'LM;WO2', 'IM;WNO', 'IM;WO1', 'IM;WO2', 'SM;WNO', 'SM;WO1', 'SM;WO2']
        labels_coord = ['00', '01', '02', '10', '11', '12', '20', '21', '22']

        metallicity_values = ['0.0001', '0.0010', '0.0200']
        overshoot_values = ['0.0000', '0.0140', '0.0160']

        # The following tag1/tag2 variables, serve as two components of the plot legend

        tag1 = str(round(float(self.getMass()), 1)) + r'M$_{\odot}$'

        # TODO: Make next clause more efficient
        #       Perhaps break down the if statement into two separate conditions, so the nested
        #       for-loop would be entered only when the right metallicity has been found

        for i in range(len(metallicity_values)):
            for j in range(len(overshoot_values)):

                if (
                        f'{self.getMetallicity()}' == metallicity_values[i] and
                        f'{self.getOvershoot()}' == overshoot_values[j]
                    ):

                    idx = labels_coord.index(f'{i}{j}')
                    tag2 = labels[idx]
                    break

        plt.plot(h.data('log_center_Rho'), h.data('log_center_T'), color = 'b', label = f'{tag1}, {tag2}')

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








    # STATIC METHODS
    # --------------------------------------------------------------------------
    @staticmethod
    def name_is_valid(name_as_string):

        '''
        This static method checks if a given string is a valid name, and
        returning a boolean True or False.
        '''

        isValid = False

        try:
            mass, metallicity, overshooting = map(str, name_as_string.split('_'))

            if len(mass) == 6 and len(metallicity) == 6 and len(overshooting) == 6:
                try:
                    a = float(mass)
                    b = float(metallicity)
                    c = float(overshooting)

                    isValid = True
                    return isValid

                except:
                    return isValid


            else:
                return isValid

        except:
            return isValid



    @staticmethod
    def export_csvFile(zip_object, name = 'csvData'):

        '''
        This static method takes a <zip object> as a mandatory argument,
        and creates a csv file of the data along with a single row header.

        The header format follows the output of the <MESA_STAR.getCoreMass()>
        method.

        The default name of the output file is "csvData" and it can be changed
        using the optional  argument "name".
        '''

        # A simple header that follows the self.getCoreMass() method
        header = ['#initial_mass', 'initial_metallicity', 'overshooting_factor', 'core_mass', 'envelope_mass']

        # List that stores all data + header
        csvData = [header]

        # Unpack data
        for a,b,c,d,e in zip_object:

            data_row = [a,b,c,d,e]
            csvData.append(data_row)


        # Create csv file
        with open(f'{name}.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)

        csvFile.close()










def main():
    help(MESA_STAR)

if __name__ == '__main__':
    main()
