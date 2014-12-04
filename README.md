neuroinf-ml
===========

A set of machine learning tools coded as exercises to learn the algorithms more 
thoroughly. Coded as a group in the DTC for Neuroinformatics at the University 
of Edinburgh.

Discrete Latent State Markov Model (aka HMM)
--------------------------------------------

Working on this through the burglar example in Barber's textbook. Have so far
implemented filtering. This can be run using the `burglar.py` script, but 
so far the result isn't plotted (Gavin: wanted to make it write an animated 
gif). However, it's easy to plot the data however you might like as it's 
returned by the `main()` function in `burglar.py`. So just start up IPython
or similar and:

```python
import burglar
filter_results = burglar.main()
```

For example, if you'd like to plot using the heatmap function we've already
written then all you need to do is:

```python
t = 1
import neuroml.plotting
neuroml.plotting.heatmap(filter_results[t])
```

Where t is the time of the hidden state you want to plot.
