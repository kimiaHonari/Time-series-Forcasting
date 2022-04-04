
import matplotlib.pyplot as plt
import numpy as np
x_value=np.zeros((16,30001))
y_value = np.zeros((16,30001))
e=0
for i in range(16,17):
    name='methane/delay_val/d'+str(i)+'.txt'
    f = open(name)
    data = f.read()
    f.close()
    lines = data.split('\n')
    lines = lines[1:]
    xvalue=[]
    yvalue=[]
    # l=0

    for line in lines:

        m=0
        for x in line.split(' '):
            if x=='':
                break
            if m==0:
                xvalue.append(int(x))
                m+=1
            else:
                yvalue.append(float(x))
                # if float(x)< (1/2.7182818285):
                #     print("value",i," : ",float(x))
                #     print("delay",i," : ",e)
                #     m=2
                #     break

        # e+=1
        # if m==2:
    print(x_value.shape)
    x_value[e]=np.array(xvalue,object)
    y_value[e]=np.array(yvalue,object)

    # l=l+1
    e+=1

x = np.linspace(0,30000,100)
y = 0*x+(1/2.7182818285)
plt.plot(x,y,linestyle='dashed',label="threshold")
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
plt.ylabel("mutual value")
plt.xlabel("delay value")
# plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

plt.show()

