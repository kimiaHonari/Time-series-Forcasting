
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

x_value=np.zeros((16,20))
y_value = np.zeros((16,20))
e=0
for i in range(1,17):
    name='co/dime/dimc'+str(i)+'.txt'
    f = open(name)
    data = f.read()
    f.close()
    lines = data.split('\n')

    xvalue=[]
    yvalue=[]
    l=0

    for line in lines:

        m=0
        for x in line.split(' '):
            if x=='':
                break
            if m==0:
                xvalue.append(int(x))
                m+=1
            elif m==1:
                yvalue.append(float(x))
                m+=2
            else:
                break

    x_value[e]=np.array(xvalue,object)
    y_value[e]=np.array(yvalue,object)
    e+=1


plt.plot(x_value[0],y_value[0],label="Senosr 1")
plt.plot(x_value[1],y_value[1],label="Senosr 2")
plt.plot(x_value[2],y_value[2],label="Senosr 3")
plt.plot(x_value[3],y_value[3],label="Senosr 4")
plt.plot(x_value[4],y_value[4],label="Senosr 5")
plt.plot(x_value[5],y_value[5],label="Senosr 6")
plt.plot(x_value[6],y_value[6],label="Senosr 7")
plt.plot(x_value[7],y_value[7],label="Senosr 8")
plt.plot(x_value[8],y_value[8],label="Senosr 9")
plt.plot(x_value[9],y_value[9],label="Senosr 10")
plt.plot(x_value[10],y_value[10],label="Senosr 11")
plt.plot(x_value[11],y_value[11],label="Senosr 12")
plt.plot(x_value[12],y_value[12],label="Senosr 13")
plt.plot(x_value[13],y_value[13],label="Senosr 14")
plt.plot(x_value[14],y_value[14],label="Senosr 15")
plt.plot(x_value[15],y_value[15],label="Senosr 16")
plt.ylabel("False Nearest Neighbor")
plt.xlabel("Dimension")
# chartBox = plt.get_position()
# plt.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
plt.show()


d1_co=[6950,0,None,None,11876,4831,None,None,7419,8654,None,None,3834,4482,None,None,(7000),25040]
d1_methane=[6954,1,12011,11877,25534,25814,13096,12292,4757,5073,12374,12399,None,None,22938,23429]
