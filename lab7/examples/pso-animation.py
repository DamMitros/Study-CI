import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher 


options = {'c1':0.5, 'c2':0.3, 'w':0.1} 
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
optimizer.optimize(fx.sphere, iters=50) 
# tworzenie animacji 
m = Mesher(func=fx.sphere) 
animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, mark=(0, 0))
animation.save('plot4.gif', writer='imagemagick', fps=10)

# W przypadku pierwotnych C1, C2, W animacja radziła sobie dobrze,
# rój trzymał się blisko siebie, ale nie był w stanie znaleźć minimum.(plot0.gif)

# Zmiana wartości C1 do ekstremalnych wartości 2.0 i C2 do 0.1 spowodowała,
# że animacja była bardziej chaotyczna, a cząstki poruszały się w różnych kierunkach.
# Jednak roj był w stanie znaleźć minimum i radził sobie lepiej. (plot1.gif)

# Odwrócenie tych wartości (C1=0.1, C2=2.0) spowodowało, że cząstki 
# podążały za sobą, ale dalej w chaotyczny sposób. (plot2.gif)

# Zmiana wartości W do 2.0 spowodowała, że cząstki uciekły poza 
# granice i nie były w stanie znaleźć minimum. (plot3.gif)

# Zmiana wartości W do 0.1 spowodowała, że cząstki poruszały się w
# najbardziej skorelowany sposób, ale nie były w stanie znaleźć minimum. (plot4.gif)
# Poszło im zdecydowanie najgorzej. (plot4.gif)

