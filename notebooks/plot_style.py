import matplotlib.pyplot as plt

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

plt.rc('text', usetex=True)                             #    commented out as requires tex
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', family='serif')
plt.rc('font', size='22')
#plt.rcParams['figure.dpi'] = 150
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1.5

plt.rcParams['lines.linewidth'] = 1.5


# set color scheme
#plt.style.use('tableau-colorblind10') --------------

#colorblind10-tableua
#colors = ["#006BA4", "#FF800E", "#ABABAB", "#595959","#5F9ED1", "#C85200", "#898989", "#A2C8EC", "#FFBC79", "#CFCFCF"]
colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7","#F0E442", "#56B4E9"] # seaborne-colorblind
#WA -0, RR -2 GR -1 - cos maybe it look nicer