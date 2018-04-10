# Recurrent Neutral Network -- RNN
# Unlike normal network, the neure of RNN is like a state machine, which has memories to some degree.

# I'd like to write a RNN with only one input, one ourput and two dimensional state.

import numpy as np

X=[1,2]
state=[0.0,0.0]

# Different weights for input and state
w_cell_state=np.asarray([[0.1,0.2],[0.3,0.4]])
w_cell_input=np.asarray([0.5,0.6])
b_cell=np.asarray([0.1,-0.1])

# Parameters for full-connected-later's output.
w_output=np.asarray([[1.0],[2.0]])
b_output=0.1

# Executing the forward process in time's order.
for i in range(len(X)):
    before_activation=np.dot(state,w_cell_state)+X[i]*w_cell_input+b_cell
    state=np.tanh(before_activation)

    final_output=np.dot(state,w_output)+b_output

    print("before activaion: ",before_activation)
    print("state: ",state)
    print("output: ",final_output)
