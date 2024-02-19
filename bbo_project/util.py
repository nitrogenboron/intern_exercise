import numpy as np
#utilities

def is_concave(f):
    n = len(f)
    for i in range(1, n - 1):
        if f[i] >= (f[i-1] + f[i+1]) / 2:
            pass 
        else:
            return False
    return True

def is_convex(f):
    n = len(f)
    for i in range(1, n - 1):
        if f[i] <= (f[i-1] + f[i+1]) / 2:
            pass 
        else:
            return False
    return True

def is_non_decreasing(f):
    n = len(f)
    for i in range(1, n):
        if f[i] < f[i-1]:
            return False
    return True

def is_non_increasing(f):
    n = len(f)
    for i in range(1, n):
        if f[i] > f[i-1]:
            return False
    return True


def valid_u(u):
  for uu in u:
      if uu % 5 == 0 and uu % 3 != 0:
          pass
      else:
          return False
  return True

def validate(func,a,b,c,d,u=[10,20,35,40,50,55,65]):
    def check_y_monotone_cvx(y: np.array, u: int, f: np.array) -> bool:
        valid_u(u)
        if u % 2 == 0:
            if not is_concave(f) or not is_non_decreasing(f):
                return False
        else:
            if not is_convex(f) or not is_non_increasing(f):
                return False
        return True
    x=np.linspace(a,b,100)
    y=np.linspace(c,d,100)
    
    #check x's property
    for yy in y:
      for uu in u: 
        f=func(x,yy,uu)
        if is_non_decreasing(f):
          pass
        else:
          print('invalid: x')
          return False
    #check y's property
    for xx in x:
      for uu in u:
        f=func(xx,y,uu)
        if uu%2==1 and is_non_decreasing(f) and is_concave(f):
          pass
        elif uu%2==0 and is_non_increasing(f) and is_convex(f):
          pass 
        else: 
            print('invalid:y')
            return False
    return True


def example_f1(x,y,u):
    f=x+x**3-np.sinc(u/5+1/2)*np.log(y)
    return f 

def psinc(u):
    return np.sin(u*np.pi)/(np.abs(u*np.pi))

def example_f2(x,y,u):
    f=x+x**3+psinc(u-30+1/2)*np.exp(-y)
    return f 

def prepare_data(func, a, b, c, d, U, N=1000,file_name='data.csv'):
    x = np.random.uniform(a, b, N)
    y = np.random.uniform(c, d, N)
    u = np.random.choice(U, N)
    f = func(x, y, u)
    p = {'x': x, 'y': y, 'u': u}
    f = f.reshape(-1, 1)
    
    data = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), u.reshape(-1, 1), f), axis=1)
    header = 'x,y,u,f'
    np.savetxt(file_name, data, delimiter=',', header=header, comments='')
    return p, f

def prepare_example_dataset1():
    a,b,c,d=0,1,1,2
    u=np.array([u for u in range(1, 300) if u % 5 == 0 and u % 3 != 0])
    N_train=10000
    N_val=2000
    p_train,f_train=prepare_data(example_f1,a,b,c,d,u,N_train,'train1.csv')
    p_val,f_val=prepare_data(example_f1,a,b,c,d,u,N_val,'val1.csv')
    return p_train,f_train,p_val,f_val
    
def prepare_example_dataset2():
    a,b,c,d=0,1,1,2
    u=np.array([u for u in range(1, 300) if u % 5 == 0 and u % 3 != 0])
    N_train=10000
    N_val=2000
    p_train,f_train=prepare_data(example_f2,a,b,c,d,u,N_train,'train2.csv')
    p_val,f_val=prepare_data(example_f2,a,b,c,d,u,N_val,'val2.csv')
    return p_train,f_train,p_val,f_val

def split_odd_even(p,f):
    p_even = {key: value[p['u'] % 2 == 0] for key, value in p.items()}
    p_odd = {key: value[p['u'] % 2 == 1] for key, value in p.items()}

    # Split f into f_even and f_odd
    f_even = f[p['u'] % 2 == 0]
    f_odd = f[p['u'] % 2 == 1]
    return p_odd,p_even,f_odd,f_even
    
def load_data(file_name):
    if file_name:
        data= np.loadtxt(file_name, delimiter=',', skiprows=1)
        x = data[:, 0]
        y = data[:, 1]
        u = data[:, 2]
        f = data[:, 3]
    p = {'x': x, 'y': y, 'u': u}
    return p,f 


def stop_con(f,f_l,f_r):
    if f_l<=f and f_r<=f:
        return True 
    else:
        return False 


def back_tracking(x_t,g_t,f_t,eta,bbo_func):
    f=bbo_func(x_t+2**eta*np.sign(g_t))
    hist=[]
    while x_t+2**eta*np.sign(g_t)<0:
        eta=eta-1
        if eta<0:
            eta=0
            break 
    
    while f-f_t<2**(eta-1)*np.abs(g_t):
        eta=eta-1
        if eta<0:
            eta=0
            break 
        f=bbo_func(x_t+2**eta*np.sign(g_t)) 
        hist+=[x_t+2**eta*np.sign(g_t)]
    x_t=x_t+2**eta*np.sign(g_t)

    return x_t,hist

def search_maxima(bbo_func,x=10,eta=10,max_iter=30):
    #init point
    x_t=x
    f_t=bbo_func(x_t)
    f_t_l=bbo_func(x_t-1)
    f_t_r=bbo_func(x_t+1)
    g_t=(f_t_r-f_t_l)/2
    history=[x_t]
    for i in range(max_iter):
        x_t,hist=back_tracking(x_t, g_t, f_t, eta, bbo_func)
        history+=hist
        f_t=bbo_func(x_t)
        f_t_l=bbo_func(x_t-1)
        f_t_r=bbo_func(x_t+1)
        g_t=(f_t_r-f_t_l)/2
        if stop_con(f_t,f_t_l,f_t_r):
            break
    return history