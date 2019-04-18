# mesa_star_class
The [mesa_star_class] defines a stellar model based on its initial mass, initial metallicity, and overshooting factor.

For more information check [MESA documentation](http://mesa.sourceforge.net/)


## Initialize and Run
In order to use the MESA_STAR class, you have to set up the path
to your MESA directory; in a terminal window type:
  > $ echo $MESA_DIR

to display the path (assuming the MESA_DIR points to the directory you
downloaded the MESA code)


You'll also need to specify a path to a location where your plots will be
saved if you choose to do so.

**Step 1** : From the command line, open a new window and run the initialize
script as follows:

  > $ python initialize.py /path/to/MESA_DIR /path/to/save/plots/


**Step 2** : You can now use the class, and its plotters to display the diagrams
you like<sup>1</sup>.

If you want to save the figure in a location other the one you specified with
the initialize script, you can do so by using the optional argument
'plot_output_dir' as shown in the example script:

  > example.py

---
<sup>1</sup>For the moment only Rho vs T plot is available
