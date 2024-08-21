import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#Get QRC dynamics (one time-step)
def QRC_dynamics(N,T,CM_0,R_0,sk,alpha_vec,beta_vec,F,CZ):
    #Define vacuum state
    CM_vac = np.eye(2*N)
    #Define U_sk gate
    U_sk = U_s(N,sk,alpha_vec,beta_vec,F,CZ)
    
    #Reservoir state for next time step
    CM_R = T*np.dot(U_sk,np.dot(CM_0,U_sk.T)) + (1-T)*CM_vac
    R_R = np.sqrt(T)*np.dot(U_sk,R_0)
    
    #State of measured pulse
    CM_meas = (1-T)*np.dot(U_sk,np.dot(CM_0,U_sk.T)) + T*CM_vac
    R_meas = np.sqrt(1-T)*np.dot(U_sk,R_0)
    #Partial trace on p-quadratures
    CM_meas,R_meas = CM_meas[0:2*N:2,0:2*N:2],R_meas[0:2*N:2]
    #Second order moments
    O2nd = CM_meas + np.dot(R_meas,R_meas.T)
    
    return CM_R,R_R,O2nd

#Iterate dynamics for a given input sequence (s_vec)
def QRC_protocol(N,T,CM_in,R_in,s_vec,alpha_vec,beta_vec):
    #Define Fourier gate and CZ chain gate
    CZ = Cz_chain(N)
    F = F_N(N)
    
    #size of the output layer
    size = int(0.5*N*(N+1))
    #readout observable matrix
    X = np.zeros((len(s_vec),size),dtype=float)
    #upper triangular indices for O2nd (second order moments matrix)
    triu_ind = np.triu_indices(N)
    for i in range(len(s_vec)):
        sk = s_vec[i]
        CM_in,R_in,O2nd = QRC_dynamics(N,T,CM_in,R_in,sk,alpha_vec,beta_vec,F,CZ)
        
        X[i,:] = O2nd[triu_ind]
    return CM_in,R_in,X

#Get performance for NARMAd task (delay d)
def get_NARMA(N,T,d,M_train=3000,M_test=500):
    #Generate random inputs from -1 to 1 (uniform distribution)
    s_washout = 2.*np.random.random(500)-1.
    s_train = 2.*np.random.random(M_train)-1.
    s_test = 2.*np.random.random(M_test)-1.
    M_washout = len(s_washout)
    
    #Generate random alpha and beta vectors (between -0.2 and 0.2)
    alpha_vec = 0.2*(2*np.random.random(N)-1.)
    beta_vec = 0.2*(2*np.random.random(N)-1.)

    #Initial state: vacuum state
    CM_in = np.eye(2*N)
    R_in = np.zeros((2*N,1),dtype=float)

    #Wash-out steps
    CM_in,R_in,_ = QRC_protocol(N,T,CM_in,R_in,s_washout,alpha_vec,beta_vec)
    #Training and test steps
    CM_in,R_in,X_train = QRC_protocol(N,T,CM_in,R_in,s_train,alpha_vec,beta_vec)
    CM_in,R_in,X_test = QRC_protocol(N,T,CM_in,R_in,s_test,alpha_vec,beta_vec)
    
    #Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #NARMA TASK
    
    #Concatenate all data
    s_all = np.concatenate((s_washout,np.concatenate((s_train,s_test))))
    #Get NARMA target function
    y_all = NARMA(s_all,d)
    #Split train and test target
    y_train,y_test = y_all[M_washout:M_washout+M_train],y_all[M_washout+M_train::]
    #Standardize target (helps normalize the MSE)
    ym,ys = np.mean(y_train),np.std(y_train)
    ys = max(ys,1e-16)
    y_train = (y_train-ym)/ys
    y_test = (y_test-ym)/ys
    #Training + get test prediction
    y_test_pred = linear_regression(X_train,y_train,X_test)
    #Get MSE
    error = nmse(y_test,y_test_pred)

    return error

#UTILITY FUNCTIONS

#Get quadratic gate for N modes
def Dq_N(N,s_val):
    input = np.zeros(2*N-1,float)
    input[0:2*N-1:2] = s_val
    Dq = np.eye(2*N)
    Dq += np.diag(input,k=-1)
    return Dq

#Get Fourier gate for N modes
def R_fun(phi):
    return np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]],dtype=float)
def F_N(N):
    F = np.zeros((2*N,2*N),dtype=float)
    for i in range(N):
        F[2*i:2*(i+1),2*i:2*(i+1)] = R_fun(np.pi/2)
    return F

#Get Cz chain for N-mode chain graph
def Cz_chain(N,strength=1):
    Cz = np.eye(2*N)
    if N>1:
        for i in range(N-1):
            Cz[2*i+1,2*(i+1)] = strength
            Cz[2*(i+1)+1,2*i] = strength
    return Cz

#Get U_sk gate (check section S3 of Supplementary Material)
def U_s(N,sk,alpha_vec,beta_vec,F,CZ):
    Dq = Dq_N(N,alpha_vec*sk+beta_vec)
    return np.dot(CZ,np.dot(F,Dq))

#Linear regression
def linear_regression(X_train, y_train,X_test):
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    return y_test_pred

#NARMAd function
def NARMA(s_all,d,alpha=0.3,beta=0.05,gamma=1.5,delta=0.1,nu=0.2,mu=0.):
    u_all = mu + nu*s_all
    y = np.zeros(len(s_all),dtype=float)
    y[0:d] = s_all[0:d]
    for i in range(d,len(s_all)):
        y_k = y[i-1]
        y[i] = alpha*y_k+beta*y_k*np.sum(y[i-d:i])+gamma*u_all[i-1]*u_all[i-d]+delta
    return y

#Normalized mean squared error
def nmse(y_target,y_pred):
    return np.mean((y_target-y_pred)**2)/max(np.mean(y_target**2),1e-16)