## Introduction to Trailblazer

This code is a collection of functions, parameters, and a solver for the class of Stochastic Integro-Differential
Equations (SIDEs) used in the memory-modeling work of Fagan, McBride, and Korolov (2023). The collection contains
all the tools to generate the simulated movement tracks used in the article.

The article: https://royalsocietypublishing.org/doi/10.1098/rsif.2022.0700

### Components

- _func_list.py_: A collection of functions to be called by the other files. Includes 
the memory and distance functions, and non-memory drift functions. Each function follows a similar template, which can be
followed to create new function types to test.
- parameter files in the _parameters_ folder. Each file defines a class _Parameter_ which is has as attributes
of all the parameters and settings used in the equation. Parameter names (alpha, beta, etc.) are the same as those used
in the paper. Functions to be used as short- and long- term memory functions, etc, are designated in this file. 
Instructions on how to designate each are listed with the individual parameters.
  - **Events** are parameters that make changes to other parameters if certain conditions are met. These are used in pigeon and
  ant models (fig_a1 and fig_a2, respectively) to either transport the animal from one point to another (in the case of
  the pigeon) or change the attracting objective (in the case of the ant) in response to the trajectory falling inside a
  particular region. Events can be _terminal_ or _non-terminal_.
  - **Plane Crossings** record when the trajectory crosses a curve described by an equation f(x,y)=0 by taking
  the value of f(x, y) at every step and recording when its sign changes.
- _side_solver.py_ The _main_ function in this file takes a given parameter set as argument and creates a weighted random
walk using those parameters, solving the SIDE described in the paper by Euler's method. Its outputs are 3 .csv files
containing:
  - The Trajectory itself
  - The time, location, and type (if multiple crossing planes given) of any plane crossings
  - The location and type of any events

### Runtime

Because the solver integrates at every step, this solver is very computationally intensive. Runs over the timeframes given
in the paper may take hours or even days, depending on the processors used. Run time for large timeframes is linearly 
correlated with two key factors
- The _timeframe_ of integration: the length of the past track to be integrated over at each step. This timeframe is
calculated and displayed at the beginning of each run. For exponentially-decaying long-term memory it is strictly inversely
correlated with the parameter k in f(t) = e^(-kt): halving k will double the integration timeframe.
- The square of the _evaluations per unit time_ n. The number of steps per unit time both increases the number of evaluations,
and increases the number of points being integrated over with each evaluative step.

_At lower levels of_ n _the solver's output can qualitatively change_. 

### Contact
Please contact Frank McBride through Github with any questions, bugs, or concerns.
