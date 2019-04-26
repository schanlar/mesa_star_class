# mesa_star_class
The ```mesa_star_class``` defines a stellar model based on its initial mass, initial metallicity, and overshooting factor.

For more information check [MESA documentation](http://mesa.sourceforge.net/)


## Download
You can download the repository as a zip file, or by typing

```bash
 ~$ git clone https://github.com/schanlar/mesa_star_class.git
```

in a terminal window.

## Initialize and Run
In order to use the MESA_STAR class, you have to set up the path
to your MESA directory; in a terminal window type:

```bash
 ~$ echo $MESA_DIR
```
to display the path (assuming the MESA_DIR points to the directory you
downloaded the MESA code)


You'll also need to specify a path to a location where your plots will be
saved if you choose to do so.

**Step 1** : From the command line, open a new window and navigate to the folder
where the ```mesa_star_class``` is located.

```bash
 ~$ cd /path/to/mesa_star_class
```

**Step 2** : When in the folder, run the initialize script as follows
(Python v3 is assumed):

```bash
 ~$ python initialize.py /path/to/MESA_DIR /path/to/save/plots/
```

**Step 3** : You can now use the class, and its plotters to display the diagrams
you like<sup>1</sup>.

If you want to save the figure in a location other the one you specified with
the initialize script, you can do so by using the optional argument
```'plot_output_dir'``` as shown in the ```example.py``` script:

  > Example
  
```python
from mesa_star_class import MESA_STAR

# Set an output path other than the one defined with the initialize script
out_path = '/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Plots'

# Make a MESA_STAR object
mesaStar = MESA_STAR('5.000', '0.0010', '0.0140',
                     history_path = '/Users/SavvasGCh/Documents/GitHub/mesa_star_class/Data/5p0M/LOGS')
                     
# Plot and save a Rho-T diagram
mesaStar.plotRhoT(saveFigure=True, figureName='Rho_vs_T_5.0000_0.0010_0.0140', plot_output_dir=out_path)
```

---
<sup>1</sup>For the moment only Rho vs T plot is available
