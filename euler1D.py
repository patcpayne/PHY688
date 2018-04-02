"""
Author: Patrick Payne
Purpose: Solve the euler equations in 1-D using a method of lines discretization
Date: 03/28/18
"""

import numpy as np
from riemann import riemann
import sys
import matplotlib.pyplot as plt

QRHO = 0
QVEL = 1
QPRES= 2

URHO = 0
UMOM = 1
UENER= 2

GAMMA = 1.4


class Grid(object):    

    def __init__(self, cells, ghosts, xmin=0.0, xmax=1.0):
    
        self.xmin = xmin
        self.xmax = xmax
        self.nx   = cells
        self.ng   = ghosts
        
        #convert from zero based to real data locations
        self.ilo = ghosts
        self.ihi = cells + ghosts - 1
        
        #physical coordinates, cell centered
        self.dx = (xmax-xmin)/cells
        self.x  = xmin + (np.arange(cells+2*ghosts)-ghosts+0.5)*self.dx
        self.xl = xmin + (np.arange(cells+2*ghosts)-ghosts)*self.dx
        self.xr = xmin + (np.arange(cells+2*ghosts)-ghosts+1.0)*self.dx
        
        self.q = self.scratchArray(3)
        self.qinit = self.scratchArray(3)
        self.u = self.scratchArray(3)        
        self.a = self.scratchArray(3)
        
    def scratchArray(self,nc=1):
        #makes a scratch array for the whole domain
        if nc == 1:
            return np.zeros((self.nx+2*self.ng), dtype=np.float64)
        else:
            return np.zeros(((self.nx+2*self.ng), nc), dtype=np.float64)
            
    def initCond(self,ic):
        # Initializes the primitive variables

        i = 0        

        if ic == "Sod's_Problem":
            self.tmax = 0.2
            while self.x[i] <= self.xmax:
                if self.x[i] < 0.5:
                    self.q[i,QRHO] = 1.0
                    self.q[i,QVEL] = 0.0
                    self.q[i,QPRES]= 1.0
                elif self.x[i] >= 0.5:
                    self.q[i,QRHO] = 0.125
                    self.q[i,QVEL] = 0.0
                    self.q[i,QPRES]= 0.1

                i += 1

        elif ic == "Double_Rarefaction":
            self.tmax = 0.15
            while self.x[i] <= self.xmax:
                if self.x[i] < 0.5:
                    self.q[i,QRHO] = 1.0
                    self.q[i,QVEL] = -2.0
                    self.q[i,QPRES]= 0.4
                elif self.x[i] >= 0.5:
                    self.q[i,QRHO] = 1.0
                    self.q[i,QVEL] = 2.0
                    self.q[i,QPRES]= 0.4

                i += 1
                
        elif ic == "Strong_Shock":
            self.tmax = 0.012
            while self.x[i] <= self.xmax:
                if self.x[i] < 0.5:
                    self.q[i,QRHO] = 1.0
                    self.q[i,QVEL] = 0.0
                    self.q[i,QPRES]= 1000.0
                elif self.x[i] >= 0.5:
                    self.q[i,QRHO] = 1.0
                    self.q[i,QVEL] = 0.0
                    self.q[i,QPRES]= 0.01

                i += 1
                
        elif ic == "Stationary_Shock":
            self.tmax = 1.0
            while self.x[i] <= self.xmax:
                if self.x[i] < 0.5:
                    self.q[i,QRHO] = 5.6698
                    self.q[i,QVEL] = -1.4701
                    self.q[i,QPRES]= 100.0
                elif self.x[i] >= 0.5:
                    self.q[i,QRHO] = 1.0
                    self.q[i,QVEL] = -10.5
                    self.q[i,QPRES]= 1.0

                i += 1
        else:
            raise ValueError("Poorly definied initial condition")
            
        self.qinit = self.q
        
    def boundCond(self,q):
        #Application of the outflow boundary condition        
        
        for n in range(self.ng):
            q[self.ilo-n-1,:] = q[self.ilo,:]
    
        for n in range(self.ng):
            q[self.ihi+1+n,:] = q[self.ihi,:]

            
def timestep(C, grid, Q, dx):
    #Calculates the time step for each iteration    

    #Calculates sound speeds 
    c = grid.scratchArray()
    c[grid.ilo:grid.ihi+1] = np.sqrt(GAMMA*Q[grid.ilo:grid.ihi+1,QPRES]) / \
        np.sqrt(Q[grid.ilo:grid.ihi+1,QRHO])

    dt = C * dx / np.amax(abs(Q[grid.ilo:grid.ihi+1,QVEL])
        +c[grid.ilo:grid.ihi+1])

    return dt
    

def prim2consv(grid,Q):
    #Converting from primitive variables to conserved variables    

    U = grid.scratchArray(3)
    
    U[:,URHO] = Q[:,QRHO]
    U[:,UMOM] = Q[:,QRHO]*Q[:,QVEL]
    U[:,UENER]= (Q[:,QPRES]/(GAMMA-1.0))+(0.5*Q[:,QRHO]*Q[:,QVEL]**(2))
    
    return U
    
def consv2prim(grid,U):
    #Converting from Conserved variables to primitive variables

    Q = grid.scratchArray(3)
    
    Q[:,QRHO] = U[:,URHO]
    Q[:,QVEL] = U[:,UMOM]/U[:,URHO]
    Q[:,QPRES]= (U[:,UENER]-0.5*U[:,UMOM]**(2)/U[:,URHO])*(GAMMA-1.0)
    
    return Q

def makeInterface(grid,Q,limiting=0):
    #builds the interface states

    ql = grid.scratchArray(3)
    qr = grid.scratchArray(3)
    q_slope = grid.scratchArray(3)

    if limiting == 1:
        for i in range(grid.ilo,grid.ihi+2):
            for j in range(QRHO,QPRES+1):
                q_slope[i,j] = minmod(Q[i+1,j]-Q[i,j],Q[i,j]-Q[i-1,j])

        for i in range(grid.ilo, grid.ihi+2):
            ql[i,:] = Q[i-1,:]+0.5*q_slope[i-1,:]
            qr[i,:] = Q[i,:]-0.5*q_slope[i,:]    
    else:
        for i in range(grid.ilo, grid.ihi+2):
            ql[i,:] = Q[i-1,:]
            qr[i,:] = Q[i,:]
            
    return ql, qr
    
    
def minmod(a,b):
    #produces the smaller slope for limiting

    
    if abs(a) < abs(b) and a*b > 0:
        val = a
    elif abs(b) < abs(a) and a*b > 0:
        val = b
    else:
        val = 0
        
    return val


def main(cells,ic,limiting):
    
    # Necessary Initializations for simulation size, initial conditions
    #   and the time step size (Courant Number)
#    cells = 256
    ghosts = 2
#    ic = "Sod's_Problem"
#    ic = "Double_Rarefaction"
#    ic = "Strong_Shock"
#    ic = "Stationary_Shock"
    courant = 0.8
    t = 0.0

    #Builds the grid
    grid = Grid(cells,ghosts)
    
    #Initializes Simulation
    grid.initCond(ic)
    grid.boundCond(grid.q)
    
    #Sets tmax to fixed value
    tmax = grid.tmax

    #Converts initialization into conserved variables
    grid.u = prim2consv(grid,grid.q)

    #Building additional arrays 
    ql = grid.scratchArray(3)
    qr = grid.scratchArray(3)
    utemp = grid.scratchArray(3)
    flux = grid.scratchArray(3)
    

    #Begin computation loop
    while t < tmax:

        #Applies boundary conditions to the grid
        grid.boundCond(grid.q)
        
        #Computes the time step dt = C * (dx/max(abs(u) + c)),
        # where c is the sound speed
        dt = timestep(courant,grid,grid.q,grid.dx)  

        #Ensures that the last time step brings the simuation
        # to tmax
        if t+dt > tmax:
            dt = tmax - t

        for i in range(0,2):
            #Assigns values for cell interfaces (NO LIMITING)
            ql, qr = makeInterface(grid,grid.q,limiting)

            #print(ql)
            #print(qr)


            #Solves riemann problem
            for i in range(grid.ilo,grid.ihi+2):
                flux[i,:] = riemann(ql[i,:],qr[i,:],GAMMA)

            #Computes the advective term (FIRST ORDER)
            for i in range(grid.ilo,grid.ihi+1):
                grid.a[i,URHO] = -1.0 * (flux[i+1,URHO] - flux[i,URHO]) / grid.dx
                grid.a[i,UMOM] = -1.0 * (flux[i+1,UMOM] - flux[i,UMOM]) / grid.dx
                grid.a[i,UENER] = -1.0 * (flux[i+1,UENER] - flux[i,UENER]) / grid.dx

            if i == 1: break
                
            utemp[:,:] = grid.u[:,:]+0.5*dt*grid.a[:,:]
            grid.q = consv2prim(grid,utemp)

        #Updates the conserved quantities
        grid.u[:,:] += grid.a[:,:]*dt

        #Converts the conserved quantities to primitive varaibles
        grid.q = consv2prim(grid,grid.u)

        # updates time
        t += dt

#        print("Primitive Variables:\n {}".format(grid.q))
#        print("\n Conserved Variables:\n {}".format(grid.u))
#        print("\n ==================================== \n")

    #NEEDS PLOTTING STUFF

    if ic == "Sod's_Problem":
        data = np.genfromtxt('sod_exact.out',  skip_header=4,
                         names=['x', 'rho', 'u', 'p', 'e'])
                         
    elif ic == "Double_Rarefaction":
        data = np.genfromtxt('double_rarefaction_exact.out',  skip_header=4,
                         names=['x', 'rho', 'u', 'p', 'e'])      
    
    elif ic == "Strong_Shock":
        data = np.genfromtxt('strong_shock_exact.out',  skip_header=4,
                         names=['x', 'rho', 'u', 'p', 'e'])
                         
    elif ic == "Stationary_Shock":
        data = np.genfromtxt('slow_shock_exact.out',  skip_header=4,
                         names=['x', 'rho', 'u', 'p', 'e'])        

    else:
        exit()

    plt.clf()
    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 2, figsize=(12,12))

    if ic == 'Strong_Shock': axarr[0,0].set_ylim([0,7])
    axarr[0, 0].plot(data['x'],data['rho'])    
    axarr[0, 0].plot(grid.x, grid.q[:,QRHO],linestyle='None',marker='+')
    axarr[0, 0].set_title('Density')
    
    axarr[0, 1].plot(data['x'],data['u'])   
    axarr[0, 1].plot(grid.x, grid.q[:,QVEL],linestyle='None',marker='+')   
    axarr[0, 1].set_title('Velocity')
    
    axarr[1, 0].plot(data['x'],data['p']) 
    axarr[1, 0].plot(grid.x, grid.q[:,QPRES],linestyle='None',marker='+')   
    axarr[1, 0].set_title('Pressure')

    u_internal_energy = grid.scratchArray()
    u_internal_energy = (grid.u[:,UENER]-0.5*grid.u[:,UMOM]**2
                        /grid.u[:,URHO])/grid.q[:,QRHO]    

    axarr[1, 1].plot(data['x'],data['e'])    
    axarr[1, 1].plot(grid.x, u_internal_energy,linestyle='None',marker='+')
    axarr[1, 1].set_title('Energy')
    
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)    
    figure = ic+str(cells)+".png"
    print(figure)
    plt.savefig(figure,format='png')
        
if __name__ == "__main__":
    
    ic = ["Sod's_Problem","Double_Rarefaction","Strong_Shock","Stationary_Shock"]
    cells = [64,128,256]
    for i in range(0,len(ic)):
        for j in range(0,len(cells)):
            main(cells[j],ic[i],limiting=1)