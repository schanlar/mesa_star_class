# Import the MESA_STAR class
# ------------------------------------------------------------------------------
from mesa_star_class import MESA_STAR as ms


# Set the path where all of the plots will be saved
# ------------------------------------------------------------------------------
out_path = "/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Plots"

# Create a MESA_STAR object
# ------------------------------------------------------------------------------
myStar = ms(
    mass="7.0000",
    metallicity="0.0200",
    overshooting="0.0000",
    history_path="/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/7p0M/LOGS",
    profile_path="/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/7p0M",
)

print(myStar)
print(myStar.__dict__)
print("*" * 30)


# Create a MESA_STAR object from a string input
# ------------------------------------------------------------------------------
new_input = "3.400_0.0010_0.0160"

myStar2 = ms.from_string(
    new_input,
    profile_number="28",
    profile_path="/vol/aibn1107/data2/schanlar",
    other_kwarg="not_valid",
)

print(myStar2)
print(myStar2.__dict__)
print("*" * 30)


# Test validity of names
# ------------------------------------------------------------------------------
t = [
    "2.5000_0.0200_0.0160",
    "asinglestring",
    "2.5000_0.0200_0.016s",
    "2.5000_0.200_0.0160",
]


for name in t:
    if ms.name_is_valid(name):
        print("Valid name!")
    else:
        print(f"problem with {name}")

print("*" * 30)


# Set paths for various models
# ------------------------------------------------------------------------------
paths = [
    "/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/2p6M/LOGS",
    "/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/5p0M/LOGS",
    "/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/7p0M/LOGS",
]

# Set the names of these models.
# In this example, the values for metallicity and overshooting were chosen arbitrarily
# The naming follows the rule: mass_metallicity_overshooting
# ------------------------------------------------------------------------------
names = ["2.6000_0.0001_0.0160", "5.0000_0.0010_0.0140", "7.0000_0.0200_0.0000"]


# Loop over the data
# ------------------------------------------------------------------------------
for name, direct in zip(names, paths):

    star = ms.from_string(name, history_path=direct)

    figureName = f"Rho_vs_T_{name}.pdf"

    star.plotRhoT(saveFigure=True, figureName=figureName, plot_output_dir=out_path)
