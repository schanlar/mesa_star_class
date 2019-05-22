'''
@version: v.22.05.19
@description: https://github.com/schanlar/mesa_star_class
'''

import numpy as np
import math
import mesa_reader as mr
import os, re, glob, csv
import logging
import astropy.units as u
import astropy.constants as c
from functools import wraps
from file_read_backwards import FileReadBackwards as frb
import matplotlib.pyplot as plt

# The basic configuration for logging file
# Uncomment this if you want to use the root logger
# logging.basicConfig(filename='mesa_star.log', level=logging.INFO,
#                    format='%(asctime)s:%(levelname)s:%(message)s')

# Create a logger other than root
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Specify the format of the logger
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

# Handler to write log data into a file, from
# debug priority and above
file_handler = logging.FileHandler('mesa_star.log')
file_handler.setFormatter(formatter)

# Handler to display log data on the screen
# from warning priority or worse
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)



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

        If there is no file with such a name, you can use the "find_profile_number()"
        method, in order to find a specific profile.

        The absolute paths for the history file, and a given profile can be set using the kwargs "history_path",
        and "profile_path" respectively.
        '''

        # For internal use
        self._mesa = 'r10398'

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

        # Make an entry in the mesa_star.log file, every time
        # a MESA_STAR instance is created
        logger.info(f'Created MESA_STAR instance; Mass: {self.initial_mass}, Metallicity: {self.initial_metallicity}, Overshooting: {self.initial_overshooting}')


    # Output
    def __str__(self):
        return 'MESA_STAR[' + '\n' + \
            f'> Initial mass: {self.initial_mass} Msol' + '\n' + \
            f'> Initial metallicity: {self.initial_metallicity}'  + '\n' + \
            f'> Overshooting factor: {self.initial_overshooting}' + '\n' + \
            ']'


    # Destructor
    def __del__(self, verbatim = False):
        if verbatim:
            print('MESA_STAR Object has been destructed!')


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


    # MESA VERSION
    # --------------------------------------------------------------------------

    @property
    def mesa(self):
        return f'MESA VERSION: {self._mesa}'

    @mesa.setter
    def mesa(self, new_version: str):
        self._mesa = new_version


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

    def getChandraMass(self):
        '''
        Returns an estimation for the Chandrasekhar mass limit
        based on the electron fraction Y_e
        '''
        try:
            p = self.getProfile()
        except Exception as e:
            logger.exception('Something went wrong while trying to load the profile!')
            raise SystemExit(e)

        average_ye = round(np.mean(p.data('ye')), 3)
        chan_mass = round(5.836 * (average_ye ** 2), 3)
        return chan_mass

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

        except FileNotFoundError as e:
            logger.exception('Could not load history file!')
            raise SystemExit(e)

        except Exception as e:
            logger.exception('Something went wrong while trying to load the history file!')
            raise SystemExit(e)

        else:
            logger.info('History file loaded succesfully!')

        finally:
            pass


        if (h.data('star_mass')[-1] != h.data('c_core_mass')[-1]) and not (math.isclose(h.data('c_core_mass')[-1], 0.0, abs_tol = 0.0)):

            initial_mass = round(float(self.getMass()),1)
            initial_metallicity = float(self.getMetallicity())
            overshooting_factor = float(self.getOvershoot())

            final_core_mass = round(h.data('c_core_mass')[-1], 3)
            final_envelope_mass = round(h.data('star_mass')[-1] - h.data('c_core_mass')[-1], 3)

            return initial_mass, initial_metallicity, overshooting_factor, final_core_mass, final_envelope_mass

        # When the carbon core mass is equal to the total mass of the star, or
        # when -due to MESA core definition- the carbon core mass is evaluated
        # to zero, we need manual investigation
        elif h.data('star_mass')[-1] == h.data('c_core_mass')[-1] or \
            math.isclose(h.data('c_core_mass')[-1], 0.0, abs_tol = 0.0):

            try:
                p = self.getProfile()

            except FileNotFoundError as e:
                logger.exception('Failed to load profile!')
                raise SystemExit(e)

            except Exception as e:
                logger.exception('Something went wrong while trying to load the profile!')
                raise SystemExit(e)

            else:
                initial_mass = round(float(self.getMass()),1)
                initial_metallicity = float(self.getMetallicity())
                overshooting_factor = float(self.getOvershoot())

                mask = 0.80 * max(p.data('logP'))
                logP = p.data('logP')
                logR = p.data('logR')
                core_boundary = p.data('mass')[np.where(logP < mask)][-1]

                print(f'Final core mass estimate: {round(core_boundary,3)} Msol')

                # Make a plot for visual aid
                fig, ax1 = plt.subplots(figsize = (13,9))

                color = 'tab:red'
                ax1.set_xlabel(r'Mass coordinate [M$_{\odot}$]')
                ax1.set_ylabel(r'$\logP$ [Ba]', color=color)
                ax1.plot(p.data('mass'), logP, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:blue'
                ax2.set_ylabel(r'$\log R$ [R$_{\odot}$]', color=color)  # we already handled the x-label with ax1
                ax2.plot(p.data('mass'), logR, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                fig.tight_layout()

                plt.axvline(core_boundary, c = 'gray', linestyle = '--')
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

                            # Make a plot for visual aid
                            fig, ax1 = plt.subplots(figsize = (13,9))

                            color = 'tab:red'
                            ax1.set_xlabel(r'Mass coordinate [M$_{\odot}$]')
                            ax1.set_ylabel(r'$\logP$ [Ba]', color=color)
                            ax1.plot(p.data('mass'), logP, color=color)
                            ax1.tick_params(axis='y', labelcolor=color)

                            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                            color = 'tab:blue'
                            ax2.set_ylabel(r'$\log R$ [R$_{\odot}$]', color=color)  # we already handled the x-label with ax1
                            ax2.plot(p.data('mass'), logR, color=color)
                            ax2.tick_params(axis='y', labelcolor=color)

                            fig.tight_layout()

                            plt.axvline(core_boundary, c = 'gray', linestyle = '--')
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

        try:
            h = self.getHistory()

        except FileNotFoundError as e:
            logger.exception('Could not load history file!')
            raise SystemExit(e)

        except Exception as e:
            logger.exception('Something went wrong while trying to load the history file!')
            raise SystemExit(e)

        else:
            logger.info('History file loaded succesfully!')


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


    @__plot_decorator
    def plotElectronFraction(self,
                             saveFigure = False,
                             plot_dir = '',
                             colour = 'blue',
                             figureName = 'mass_vs_ye.pdf'):

        info = self.getName()
        try:
            p = self.getProfile()
        except Exception as e:
            logger.exception('Something went wrong while trying to load the profile!')
            raise SystemExit(e)

        plt.figure(figsize = (13,9))
        plt.xlabel(r'Mass coordinate [M$_{\odot}$]', size = 15)
        plt.ylabel(r'Y$_e$', size = 15)

        if saveFigure:

            plt.plot(p.data('mass'), p.data('ye'), c = colour)
            plt.savefig(f'{figureName}', bbox_inches = 'tight', dpi = 300)
            plt.clf()
        else:

            plt.plot(p.data('mass'), p.data('ye'), c = colour)




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
                    logger.exception(f'The name {name_as_string} is not valid!')
                    return isValid

            else:
                logger.warning(f'The length of the name {name_as_string} is not valid!')
                return isValid

        except:
            logger.exception(f'The name {name_as_string} is not valid!')
            return isValid



    @staticmethod
    def export_csvFile(zip_object, name = 'csvData', termination=False):

        '''
        This static method takes a <zip object> as a mandatory argument,
        and creates a csv file of the data along with a single-row header.

        The header format follows the output of the <MESA_STAR.getCoreMass()>
        method.

        If termination=True, the csv file will include another column that
        will store the termination code for the stellar model.

        The default name of the output file is "csvData" and it can be changed
        using the optional  argument "name".
        '''

        # A simple header that follows the self.getCoreMass() method
        if termination:
            header = ['#initial_mass', 'initial_metallicity', 'overshooting_factor', 'core_mass', 'envelope_mass', 'termination_code']
        else:
            header = ['#initial_mass', 'initial_metallicity', 'overshooting_factor', 'core_mass', 'envelope_mass']

        # List that stores all data + header
        csvData = [header]

        # Unpack data
        if termination:
            for a,b,c,d,e,f in zip_object:

                data_row = [a,b,c,d,e,f]
                csvData.append(data_row)
        else:
            for a,b,c,d,e in zip_object:

                data_row = [a,b,c,d,e]
                csvData.append(data_row)


        # Create csv file
        with open(f'{name}.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)

        csvFile.close()


    @staticmethod
    def find_profile_number(path: str, num = -1):
        '''
        Returns a string for the profile number.

        It takes a mandatory argument for the path where
        the profiles.index file is located.

        The optional argument "num" defines the number of row
        in the file.

        num < 0 means the script will start reading the
        profiles.index file bottop-up, and will stop at the
        specified row.

        Thus, num = -1 will return the last saved profile,
        num = -2 will return the second to last etc.

        num > 0 means the script will start reading the
        profiles.index file up-bottom, and will stop at the
        specified row.

        Thus, num = 1 will return the first saved profile
        within the profiles.index file.

        num = 0 will print the header info of the profiles.index
        file. In this case, the funtion does not return anything.
        '''

        counter = 0

        if num < 0:
            with frb(f'{path}/profiles.index') as file:

                for line in file:
                    counter -= 1

                    if counter == num:

                        info = re.split(' |, |\n', line)

                        # Try casting the profile number as integer
                        try:
                            info[-1] = int(info[-1])
                        except:
                            info[-1] = int(info[-2])

                        #print(f'The profile is: profile{info[-1]}.data')
                        return repr(info[-1])

                if abs(num) > counter:
                    raise ValueError('Number out of range!')

        elif num > 0:
            with open(f'{path}/profiles.index', 'r') as file:

                # Skip the first row
                next(file)

                for line in file:
                    counter += 1

                    if counter == num:

                        info = re.split(' |, |\n', line)

                        # Try casting the profile number as integer
                        try:
                            info[-1] = int(info[-1])
                        except:
                            info[-1] = int(info[-2])

                        #print(f'The profile is: profile{info[-1]}.data')
                        return repr(info[-1])

                if abs(num) > counter:
                    raise ValueError('Number out of range!')

        else:
            with open(f'{path}/profiles.index', 'r') as file:
                for line in file:
                    print(f'Header info: {line}')
                    break


    @staticmethod
    def find_termination_code(path_to_file: str):
        '''
        The function takes a single argument which is the path to the
        output file, e.g. condor.out, and returns the termination code
        for this model (if it exists).
        '''

        keyword = 'termination code: '
        line_number = 0

        with frb(path_to_file) as file:

            for line in file:
                line_number += 1

                if line.startswith(keyword) and line_number <= 50:

                    code = line.split(' ')

                    return code[-1]

                elif line_number > 50:
                    out = 'N/A'

                    return out














def main():
    help(MESA_STAR)

if __name__ == '__main__':
    main()
