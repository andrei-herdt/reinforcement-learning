Exercise 2.1
The probability is 0.5 since the probability of selecting the non-greedy action is set to 0.5 by the epsilon parameter.

Exercise 2.2
| t | A_t | R_t | Q(A_t)        | next argmax Q(A) | greedy or exploratory? |
| 1 | 1   | -1  | -1            | 2 or 3 or 4      | na                     |
| 2 | 2   | 1   | 1/1=1         | 2                | maybe exploratory      |
| 3 | 2   | -2  | 1+(-2)/2=-1/2 | 3 or 4           | greedy                 |
| 4 | 2   | 2   | (-1+2)/3=1/3  | 2                | definitely exploratory |
| 5 | 3   | 0   | 0             |                  | definitely exploratory |

Exercise 2.3
In the long run, the method with the smaller epsilon above zero will perform better in selecting the best action.
One method will deviate from the optimal action in 10% of cases while the other will do so only every 100th step in average.
