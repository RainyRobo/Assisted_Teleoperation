import jax
import numpy as np
import einops
# batch_shape =28 
rng = jax.random.PRNGKey(0)
# time_rng, noise_rng = jax.random.split(rng)

# actions = jax.random.normal(rng, (28, 50, 32))
# noise = jax.random.normal(noise_rng, actions.shape) # noise (28, 50, 32)

# time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001 #(28,)
# print(time.shape)
# time_expanded = time[..., None, None]
# print(time_expanded.shape)

# x_t = time_expanded * noise + (1 - time_expanded) * actions

# u_t = noise - actions
# a = np.array([[[1, 2, 3], [4, 5, 6]]])
# print(a.shape)
# b = np.array([[[13, 14, 15], [16, 17, 18]]])
# c = np.concatenate((a, b), axis=-1)
# print(c)
# print(c.shape)
# d=[a,b]
# print("d",d)
# print("d shape",np.array(d).shape)

def cond(carry):
    x_t, time = carry
    print("time: ",time)
    print( "-dt / 2: ", -dt / 2)
    # robust to floating-poin
    return time >= -dt / 2 # 


dt = -0.1
time =1.0

noise = jax.random.normal(rng, (1, 50, 32))

for i in range(30):
    print(i)
    a =(noise, time)
    b = cond(a)
    print(b)
    time += dt
    




