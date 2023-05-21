"Question","Answer"
"What are the two main ideas that dynamic programming relies on?","Dynamic programming relies on two main ideas: (1) breaking down the problem into subproblems (Bellman equation) and (2) solving the subproblems iteratively."
"Define the Bellman equation for value function `v_pi`.","The Bellman equation for value function `v_pi` is `v_pi(s) = Σ_a π(a|s) Σ_p(s',r|s,a)[r + γv_pi(s')]` for all `s`."
"Define the Bellman optimality equation for value function `v*`.","The Bellman optimality equation for value function `v*` is `v*(s) = max_a Σ_p(s',r|s,a)[r + γv*(s')]` for all `s`."
"What is policy evaluation (prediction problem) in dynamic programming?","Policy evaluation is the problem of determining the state-value function `v_pi` for a given policy `π`."
"What is policy improvement in dynamic programming and how does it lead to better or equal policies?","Policy improvement is the process of making a new policy that improves on an original policy, by making it greedy with respect to the value function of the original policy. It leads to better or equal policies because, for all `s` in `S`, `q_pi(s, π'(s)) >= v_pi(s)`."
"What is policy iteration in dynamic programming?","Policy iteration is a method in dynamic programming that consists of two steps: policy evaluation and policy improvement. It's an iterative method that alternates between these two steps until convergence."
"What is value iteration and how does it differ from policy iteration?","Value iteration is a method in dynamic programming that combines policy improvement and truncated policy evaluation into a single update. It's an iterative method that estimates `v*` and `π*` at the same time."

