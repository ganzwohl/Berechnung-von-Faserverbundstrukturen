import numpy as np



def CLT(E1,E2, nu_12, G12):
    print('1.计算刚度矩阵')
    nu_21 = nu_12 *E2/E1
    Qij = np.array([[E1 / (1 - nu_12 * nu_21), nu_12 * E2 / (1 - nu_12 * nu_21), 0],
                    [nu_12 * E2 / (1 - nu_12 * nu_21), E2 / (1 - nu_12 * nu_21), 0], [0, 0, G12]])
    print(np.around(Qij,decimals=2))
    print('单位， MPa， N/mm**2')

    print('2.计算ABD矩阵')
    hk = h/len(t)
    z0 = -h/2
    A = np.zeros((3,3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for i in np.arange(1,len(t)+1):
        tr = t[i-1] * np.pi /180
        T = np.array([[(np.cos(tr)) ** 2, (np.sin(tr)) ** 2, -2 * np.cos(tr) * np.sin(tr)],
                       [(np.sin(tr)) ** 2, (np.cos(tr)) ** 2, 2 * np.cos(tr) * np.sin(tr)],
                       [np.cos(tr) * np.sin(tr), -np.cos(tr) * np.sin(tr), (np.cos(tr)) ** 2 - (np.sin(tr)) ** 2]])
        Qij_ = np.dot(np.dot(T,Qij),T.T)
        zi = i*hk +z0
        z_i = 0.5*(zi + zi -hk)
        A += Qij_ * hk
        B += Qij_ * z_i * hk
        D += Qij_ * (z_i ** 2 + 1 / 12 * hk ** 2) * hk
    print(np.around(A,decimals=1)) #N/mm
    print(np.around(B,decimals=1)) #N
    print(np.around(D,decimals=1)) #N*mm
    ABD_Matrix = np.vstack((np.hstack((A, B)), np.hstack((B, D))))

    print('3.计算等效模量')
    Ex_A = 1/np.linalg.inv(ABD_Matrix)[0,0]/h
    Ey_A = 1 / np.linalg.inv(ABD_Matrix)[1, 1] / h
    Gxy_A = 1 / np.linalg.inv(ABD_Matrix)[2, 2] / h
    vxy_A = -np.linalg.inv(ABD_Matrix)[1, 0] / np.linalg.inv(ABD_Matrix)[0, 0]
    vyx_A = -np.linalg.inv(ABD_Matrix)[0, 1] / np.linalg.inv(ABD_Matrix)[1, 1]
    print("Ex: ", np.around(Ex_A, decimals=1), ", Ey: ", np.around(Ey_A, decimals=1),
          ", Gxy: ", np.around(Gxy_A, decimals=1), ", vxy: ", np.around(vxy_A, decimals=2),
          ", vyx: ", np.around(vyx_A, decimals=2))

    print('4.计算层合板变形')
    zaihe=np.array([N_M[i] for i in range(len(N_M))])
    bianxing=np.dot(np.linalg.inv(ABD_Matrix),zaihe)
    print(np.around(bianxing,decimals=6))
    epsilon0 = np.array(bianxing[0:3])
    kappa = np.array(bianxing[3:6])

    print('5.计算单层应力向量')
    sigma = []
    for i in range(1,len(t)+1):
        tr = t[i-1]*np.pi/180
        Ti = np.array([[(np.cos(tr)) ** 2, (np.sin(tr)) ** 2, -2 * np.cos(tr) * np.sin(tr)],
                       [(np.sin(tr)) ** 2, (np.cos(tr)) ** 2, 2 * np.cos(tr) * np.sin(tr)],
                       [np.cos(tr) * np.sin(tr), -np.cos(tr) * np.sin(tr), (np.cos(tr)) ** 2 - (np.sin(tr)) ** 2]])
        Qij_ = np.dot(np.dot(Ti, Qij), Ti.T)
        z_up = z0+i*hk
        z_down = z_up -hk
        sigma_up_glo = np.dot(Qij_,(epsilon0+z_up*kappa))
        sigma_down_glo = np.dot(Qij_, (epsilon0 + z_down * kappa))
        sigma_up_lok = np.dot(np.linalg.inv(Ti),sigma_up_glo)
        sigma.append(sigma_up_lok)
        sigma_down_lok = np.dot(np.linalg.inv(Ti), sigma_down_glo)
        # print('全局坐标上端应力 Mpa')
        # print(np.around(sigma_up_glo))
        # print('全局坐标下端应力 Mpa')
        # print(np.around(sigma_down_glo))
        # print('局部坐标上端应力 Mpa')
        # print(np.around(sigma_up_lok))
        # print('局部坐标上端应力 Mpa')
        # print(np.around(sigma_down_lok))
    sigma = np.asarray(sigma)
    for i in range(1,int(n/2)+1):
        print("第%i.层应力向量" % i)
        print(np.around(sigma[i-1],decimals=1))
    return sigma

def Tsai_Wu(Rt1,Rc1,Rt2,Rc2,R12,sigma,F12):
    print('6. Tsai-Wu 标准：')
    F1 = 1 / Rt1 - 1 / Rc1
    F11 = 1 / Rt1 / Rc1
    F2 = 1 / Rt2 - 1 / Rc2
    F22 = 1 / Rt2 / Rc2
    F66 = 1 / R12 ** 2
    for i in range(int(n/2)):
        Q = F11*sigma[i,0]**2+F22*sigma[i,1]**2+F66*sigma[i,2]**2
        L = F1 * sigma[i,0]+F2*sigma[i,1]
        f_res = -L / 2 /Q + (L**2/4/Q**2+1/Q)**0.5
        print('Reserve factor_Tsai Wu在第%i层是'%(i+1),np.around(f_res,decimals=2))

def HashinPuck(R12,Rt2,Rc2,sigma,p12_plus,p12_minus):
    print('7.Hashin Puck')
    R22_A = R12/2/p12_minus*((1+2*p12_minus*Rc2/R12)**0.5-1)
    p22_minus = p12_minus*R22_A/R12
    tau21_c = R12*(1+2*p22_minus)

    if sigma[1]>=0:
        F = ((sigma[2]/R12)**2+(1-p12_plus*Rt2/R12)**2*(sigma[1]/Rt2)**2)**0.5+p12_plus*sigma[1]/R12
        f_Res = 1 / F
        print('Modus A,Reserve factor:',np.around(f_Res,decimals=3))
        print('断裂角 0 度')
    elif 0 >= sigma[1] and 0 <= np.absolute(sigma[1] / sigma[2]) <= R22_A / np.absolute(tau21_c):
        F = 1 / R12 * ((sigma[2] ** 2 + (p12_minus * sigma[1]) ** 2) ** 0.5 + p12_minus * sigma[1])
        f_Res = 1 / F
        print('Modus B,Reserve factor:', np.around(f_Res, decimals=3))
        print('断裂角 0 度')
    elif 0 >= sigma[1] and 0 <= np.absolute(sigma[2] / sigma[1]) <= np.absolute(tau21_c) / R22_A:
        F = ((sigma[2] / 2 / (1 + p22_minus) / R12) ** 2 + (sigma[1] / Rc2) ** 2) * Rc2/ (-sigma[1])
        f_Res = 1 / F
        print('Modus C,Reserve factor:', np.around(f_Res, decimals=3))
        if f_Res <=1:
            BW = (R22_A / (-sigma[1])) ** 0.5
            Degree = np.degrees(np.arccos(BW))
            print('断裂角', np.around(Degree, decimals=3))
        else:
            print('未失效')

def Cuntze(Rt1,Rc1,Rt2,Rc2,R12,B,m,sigma,E1,E2,nu_12):
    print('8. Cuntze')
    epsilon1 = sigma[0]/E1-nu_12*sigma[1]/E2
    FF1 = (epsilon1+np.abs(epsilon1))*E1/2/Rt1
    FF2 = (-sigma[0]+np.abs(sigma[0]))/2/Rc1
    IFF1 = (sigma[1]+np.abs(sigma[1]))/2/Rt2
    IFF2 = np.abs(sigma[2])/(R12-B*sigma[1])
    IFF3 = (-sigma[1]+np.abs(sigma[1]))/2/Rc2
    F = (FF1**m+FF2**m+IFF1**m+IFF2**m+IFF3**m)**(1/m)
    f_Res = 1/F
    print('Cuntze, fres=',np.around(f_Res,decimals=4))



# Input data
t = [0,90,90,0] # 单层的角度
h = 1 # 总厚度，1mm
N_M = [100, 0, 0, 0, 0, 0]
n = len(t)
# Tsai-Wu
F12 = 0
# HashinPuck
p12_plus = p12_minus = 0.3
sigma_1 = np.array([0,15,10])
sigma_2 = np.array([0,-10,50])
sigma_3 = np.array([0,-100,40])
# Cuntze
B=0.3
m=3
# 强度（MPa）
Rt1=2300
Rc1=1000
Rt2=25
Rc2=130
R12=45
# 刚度 （MPa）
E1 = 147000 #181000
E2 = 9000 #10300
nu_12 = 0.31 #0.28
G12 = 7170
#sigma = CLT(E1,E2,nu_12,G12)
#Tsai_Wu(Rt1,Rc1,Rt2,Rc2,R12,sigma,F12)
HashinPuck(R12,Rt2,Rc2,sigma_1,p12_plus,p12_minus)
HashinPuck(R12,Rt2,Rc2,sigma_2,p12_plus,p12_minus)
HashinPuck(R12,Rt2,Rc2,sigma_3,p12_plus,p12_minus)
Cuntze(Rt1,Rc1,Rt2,Rc2,R12,B,m,sigma_1,E1,E2,nu_12)
Cuntze(Rt1,Rc1,Rt2,Rc2,R12,B,m,sigma_2,E1,E2,nu_12)
Cuntze(Rt1,Rc1,Rt2,Rc2,R12,B,m,sigma_3,E1,E2,nu_12)



