'''
@author Savvas Chanlaridis
@version v.09.04.19
'''


import mesa_reader as mr
import os


class MESA_STAR(object):

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

    # Output

    def __str__(self):
        return 'MESA_STAR[' + '\n' + \
            f'> Initial mass: {self.initial_mass} Msol' + '\n' + \
            f'> Initial metallicity: {self.initial_metallicity}'  + '\n' + \
            f'> Overshooting factor: {self.initial_overshooting}' + '\n' + \
            ']'

    # Accesser Methods (getters)

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


    # Mutator Methods (setters)

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




def main():
    help(MESA_STAR)

if __name__ == '__main__':
    main()
