
Tutorial 1
> import matplotlib.pyplot as plt 
    This line imports the integral pyplot. Importing pyplot as plt is a traditional standard.
> plt.plot([1,2,3],[4,5,6])
    This means, we have 3 coordinates according to the list: 1,4  2,5  3,6
> plt.show()
    As always in graphics, after drawing the complete graph in back we are bringing it to the front.

Tutorial 2
> import matplotlib.pyplot as plt
  x = [1,2,3]
  y = [5,6,7]
  x2 = [1,2,3]
  y2 = [10,11,12]
  plt.plot(x, y, label='Firt line')     #label are used to display legend of the plot
  plt.plot(x2, y2, label='Second line')
  plt.xlabel('Plot Number')
  plt.ylabel('Important Var')
  plt.title('Interesting Graph\nCheck it out')
  plt.legend()       #invoking the default legend
  plt.show()
