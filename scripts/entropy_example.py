from math import log
p1, p2 = 0.5, 0.5
q1, q2 = 0.99, 0.01
print('info lost using q to model p:', p1 * log(p1 / q1) + p2 * log(p2 / q2))
print('info lost using p to model q:', q1 * log(q1 / p1) + q2 * log(q2 / p2))