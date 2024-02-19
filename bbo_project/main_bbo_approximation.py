from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from airt.keras.layers import MonoDense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from util import *

#%%
#create the network for odd u
inputs = {name: Input(name=name, shape=(1,)) for name in list("xyu")}
outputs = MonoDense.create_type_2(
    inputs,
    units=128,
    final_units=1,
    activation="relu",
    #final_activation="softmax",
    n_layers=4,
    dropout=0.2,
    monotonicity_indicator=dict(x=1,y=1,u=0),
    is_convex=dict(x=False,y=False,u=False),
    is_concave=dict(x=False,y=True,u=False),
)
model_odd = Model(inputs=inputs, outputs=outputs)

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10_000 ,
    decay_rate=1,
)
optimizer = Adam(learning_rate=lr_schedule)
model_odd.compile(optimizer=optimizer, loss="mse")


#create the network for even u
inputs = {name: Input(name=name, shape=(1,)) for name in list("xyu")}
outputs = MonoDense.create_type_2(
    inputs,
    units=128,
    final_units=1,
    activation="relu",
    #final_activation="softmax",
    n_layers=4,
    dropout=0.2,
    monotonicity_indicator=dict(x=1,y=-1,u=0),
    is_convex=dict(x=False,y=True,u=False),
    is_concave=dict(x=False,y=False,u=False),
)
model_even = Model(inputs=inputs, outputs=outputs)

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10_000 ,
    decay_rate=1,
)
optimizer = Adam(learning_rate=lr_schedule)
model_even.compile(optimizer=optimizer, loss="mse")

#%%generate data
p_train,f_train,p_val,f_val=prepare_example_dataset1()

#split into odd and even u cases
odd_p_train,even_p_train, odd_f_train,even_f_train=split_odd_even(p_train,f_train)
odd_p_val,even_p_val, odd_f_val,even_f_val=split_odd_even(p_val,f_val)
#%%
history_odd=model_odd.fit(
    x=odd_p_train, y=odd_f_train, batch_size=32, validation_data=(odd_p_val, odd_f_val), epochs=50
)

#%%
# Plot the training loss and validation loss
plt.plot(history_odd.history['loss'][1:], label='Training Loss (Odd)')
plt.plot(history_odd.history['val_loss'][1:], label='Validation Loss (Odd)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history_odd.png')

plt.show()

#%%
history_even=model_even.fit(
     x=even_p_train, y=even_f_train, batch_size=32, validation_data=(even_p_val, even_f_val), epochs=50
 )

#%%
# Plot the training loss and validation loss
plt.plot(history_even.history['loss'][1:], label='Training Loss (Even)')
plt.plot(history_even.history['val_loss'][1:], label='Validation Loss (Even)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history_even.png')
plt.show()