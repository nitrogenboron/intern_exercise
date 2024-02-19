from util import *
import numpy as np
import matplotlib.pyplot as plt

def example_bbo_func(x,y,u):
    #the original blackbox function
    #f=np.power(x,3)+(psinc(u-180+1/2))*np.exp(-y)
    f=np.power(x,3)-(psinc(u-180+1/2)+psinc(u-60+1/2))*np.log(y)
    return f 
def h_u(u,bbo_func= example_bbo_func,a=0,b=1,c=1,d=2):
    x=b
    y=c if u%2==0 else d
    f=bbo_func(x,y,u)
    return f

def smooth_h_u(i,bbo_func= example_bbo_func,a=0,b=1,c=1,d=2):
    #this function smooth the black box function by a continuous window of size 4
    ui=convert_i2u(i,flag=False)
        
    f=[h_u(u,bbo_func= example_bbo_func) for u in ui]
    
    return max(f)

def convert_i2u(i,flag=True):#flag indicates whether return the maximum u, or the u group
    if i==0:
        ui=[5]
    else:
        ui=[30*i-20,30*i-10,30*i-5,30*i+5]
    f=[h_u(u,bbo_func= example_bbo_func) for u in ui]
    i=np.argmax(f)
    if flag==True:
        return ui[i]
    else:
        return ui



assert convert_i2u(1,flag=False)==[10,20,25,35], "Should be 30"
assert convert_i2u(2,flag=False)==[40,50,55,65], "Should be 30"

#%%
if validate(example_bbo_func, 0, 1, 1, 2):
    print('Valid')
else:
    print('Error')
u_values=np.array([u for u in range(1, 500) if u % 5 == 0 and u % 3 != 0])
h_values=np.array([h_u(u) for u in u_values])

u_search_history=search_maxima( bbo_func=h_u,x=40,eta=10,max_iter=20)
h_search_history=np.array([h_u(u) for u in u_search_history])

i_search_history=search_maxima( bbo_func=smooth_h_u,x=10,eta=10,max_iter=20)
smooth_u_search_history=[convert_i2u(i) for i in i_search_history]
smooth_h_search_history=np.array([smooth_h_u(i) for i in i_search_history])


#%%plot the search history


plt.plot(u_values,h_values,'x-.')
plt.plot(u_search_history,h_search_history,'o-')
plt.plot(smooth_u_search_history,smooth_h_search_history,'d-')
plt.xlabel('u')
plt.ylabel('h(u)')
plt.title('h(u)')
plt.legend(['h(u)','search over h(u)','search over smoothed h(u)'])
plt.savefig('search_history_u.png')
plt.show()

