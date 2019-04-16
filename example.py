from mesa_star_class import MESA_STAR
import mesa_star_plotters as msp


# Set the path where all of the plots will be saved

out_path = '/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Plots'

# Create a MESA_STAR object

myStar = MESA_STAR(mass = '7.0000',
    metallicity = '0.0200',
    overshooting = '0.0000',
    history_path = '/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/7p0M/LOGS',
    profile_path = '/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/7p0M')

print(myStar)
print(myStar.__dict__)

# Create a MESA_STAR object from a string input

new_input = '3.400_0.0010_0.0160'

myStar2 = MESA_STAR.from_string(new_input,
    profile_number = '28',
    profile_path = '/vol/aibn1107/data2/schanlar',
    other_kwarg = 'not_valid')

print(myStar2)
print(myStar2.__dict__)


# Set paths for various models

paths = ['/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/2p6M/LOGS',
        '/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/5p0M/LOGS',
        '/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/7p0M/LOGS']

# Set the names of these models.
# In this example, the values for metallicity and overshooting were chosen arbitrarily

# The naming follows the rule: mass_metallicity_overshooting

names = ['2.6000_0.0001_0.0160', '5.0000_0.0010_0.0140', '7.0000_0.0200_0.0000']

# Loop over the data

for i in range(len(paths)):

    info = names[i].split('_') # or just use the from_string() attribute
    mesastar = MESA_STAR(mass = info[0], metallicity = info[1], overshooting = info[2], history_path=paths[i])

    figureName = 'Rho_vs_T_' + names[i] + '.pdf'

    msp.plotRhoT(mesastar, saveFigure=True, figureName=figureName, plot_output_dir=out_path)
