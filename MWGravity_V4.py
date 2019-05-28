 #This program aims at solving the full Morse-Witten equations for the case of a bubble trapped inside a square cappilary, with upwards forces exerted by neighboring bubbles.
#The equations have been given in Ginot et al., Soft Matter, 2019 (DOI: 10.1039/c8sm02477d). From now on, the equation numbers will refer to the ones in this article for convenience.
import numpy as np
import matplotlib.pyplot as plt

def log(x): #just a function to help with the logarithms
	if x==0:
		return 0
	else:
		return np.log(x)


#First step is to solve Morse-Witten equations for the case of negligible gravity (that is, Bond number equal to zero). The corresponding values can be found either by an autocoherent loop to solve the equation 14-15 by inputing the full deformation values, or by using the analytical solution of eq. 16 by input of the deformation induced by the capillary wall and the force of the neighboring bubbles.
#Notice that, for a sufficiently small convergence threshold, the values found by the two methods will be in very good agreement with each others. Moreover, as they are only the input of a latter autocoherent loop, this tiny discrepancy will not affect the final result significantly enough to give us any trouble.

#Gives the force values for defined deformations, in absence of gravity
#Uses autocoherent method by matrix inversion, see eqs. 14-15
def loop_point(Xb,Xc):#Input is the normalised deformation as defined in eqs. 7-8
	fb,fc=1.2,1.2 #Arbitrary starting values for the autocherent loop
	X = np.matrix([[Xb],[Xc]],dtype='float') #Matrix rewriting of the deformations
	A11 = np.log(fb/8/np.pi)/4/np.pi #Matrix elements from eqs. 14-15
	A12 = 1/2/np.pi
	A21 = 1/4/np.pi
	A22 = np.log(fc*np.exp(1)/8/np.pi)/4/np.pi
	A = np.matrix([[A11,A12],[A21,A22]],dtype='float') #Matrix rewriting of the force prefactors	
	F = np.dot(A.I,X) #Matricial product to get the normalised forces corresponding to the given normalised deformations
	while np.abs(fb-F[0,0])/fb >= 1e-8: #Repeats until convergence is reached
		fb = F[0,0]
		fc = F[1,0]
		if fb <= 0 or fc <=0: #Forbidding negative forces (unphysical in absence of adhesion)
			F[0,0],F[1,0] = None,None
			break
		A11 = np.log(fb/8/np.pi)/4/np.pi #Update of element values
		A22 = np.log(fc*np.exp(1)/8/np.pi)/4/np.pi
		A = np.matrix([[A11,A12],[A21,A22]],dtype='float') #Update of matrix
		F = np.dot(A.I,X) #Update of matricial product
	return F[0,0],F[1,0] #Return Neighbor and Capillary forces





#Gives the force values for defined deformations, in absence of gravity
#The width deformation is fixed by Xc
#The length deformation is a range, called Range_Xb, so as to allow to screen for various longitudinal deformations
def loop_simple(Range_Xb,Xc): 
	Fb,Fc = [None]*len(Range_Xb),[None]*len(Range_Xb)
	for i in range(len(Range_Xb)):
		Fb[i],Fc[i] = loop_point(Range_Xb[i],Xc)
	return Fb,Fc #Return the forces for every value of (xb,xc), with xc always the same



#The found values are then used as an input into the loop to solve the system of equations 30-34
#There's multiple ways to solve it. Here, it is solved by using the system of equations composed from eq. (36.1) and eqs. (36.2)-2*(36.3), which presents the advantage of having a 2x2 matrix

def f_A22(Input, Bo,alpha): #Compute the prefactor of fcb in (36.2)-2*(36.3)
	Delta = 8.*np.pi*np.exp(-5./6.) #The variable Delta is defined like this in the article
	return np.log((Input+4*np.pi*Bo*np.cos(alpha)/3)/Delta)/4/np.pi+np.log(Input/Delta)/4/np.pi - 5/12/np.pi



#Compute the values of fcv, fcb and fct for defined neighbor force fbm, length deformation Xcv, Bond number and alpha angle
def CapillaryForce(fb,Fc,Xcv,Bo,alpha):
	Delta = 8.*np.pi*np.exp(-5./6.)#The variable Delta is defined like this in the article
	fcv,fcb = Fc,Fc #Input values are the ones computed without gravity by the first part of the program
	A11 = np.log(fcv/Delta)/4/np.pi-5/24/np.pi #Compute the r.h.s. matrix elements
	A12 = 1/4./np.pi
	A21 = 1/2./np.pi
	A22 = f_A22(fcb,Bo,alpha)
	A = np.matrix([[A11,A12],[A21,A22]],dtype='float')#Matrix rewriting of the force prefactors
	B1 = Xcv -Bo*np.cos(alpha)/6-fb/4/np.pi #Compute the l.h.s. matrix elements
	B2 = 2*Xcv+5.*Bo*np.cos(alpha)/18.-fb/2/np.pi-Bo*np.cos(alpha)/3*np.log((fcb +4*np.pi*Bo*np.cos(alpha)/3)/Delta)
	B = np.matrix([[B1],[B2]],dtype='float')#Matrix rewriting
	F = np.dot(A.I,B)#Matricial product of (A^-1) and B
	while np.abs(fcv-F[0,0])/fcv >=1e-14: #loops until convergence is reached
		fcv,fcb=F[0,0],F[1,0]
		if fcb <=0 : #Forbidding negative forces (unphysical in absence of adhesion)
			F[0,0],F[1,0] = None, None
			break
		A11 = np.log(fcv/Delta)/4/np.pi-5/24/np.pi#Update the matrix terms containing the forces
		A22 = f_A22(fcb,Bo,alpha)
		B1 = Xcv -Bo*np.cos(alpha)/6-fb/4/np.pi
		B2 = 2*Xcv+5.*Bo*np.cos(alpha)/18.-fb/2/np.pi-Bo*np.cos(alpha)/3*np.log((fcb +4*np.pi*Bo*np.cos(alpha)/3)/Delta)
		B = np.matrix([[B1],[B2]],dtype='float')#Matrix rewriting
		A = np.matrix([[A11,A12],[A21,A22]],dtype='float')
		F = np.dot(A.I,B) #Matricial product of (A^-1) and 
#Function returns fcv,fcb and fct, the last one being deduced from fcb by eq. 28
	return F[0,0],F[1,0],F[1,0]+4*np.pi*Bo*np.cos(alpha)/3


#Compute the bubble length LB/2R0 using eq. 26 and the explicit expressions of xbb and xbt using eqs. 30-31
def LongitudinalLength(Fbt,Fbb,Fcv,Fct,Fcb):
	if Fbb <0: #Forbidding negative forces (unphysical in absence of adhesion)
		return None
	else:
		Delta = 8.*np.pi*np.exp(-5./6.)#The variable Delta is defined like this in the article
#Use the explicit value of the Green function G(theta) for theta=pi and theta=pi/2
		if Fbt!=0:			
			a=Fbt*np.log(Fbt/Delta)/4/np.pi
		else:
			a=0
		if Fbb!=0:
			b=Fbb*np.log(Fbb/Delta)/4/np.pi
		else:
			b=0
		c = Fcv/2./np.pi+Fct/4./np.pi+Fcb/4/np.pi
		d = -5.*Fbb/24./np.pi
		e = -5.*Fbt/24./np.pi
		return 1+(a+b+c+d+e)/2.

#Compute the full force system for given deformations (longitudinal values with Range_Xb, unique lateral value for C, Bond number Bo and tilting angle alpha formed with horizontal axis)
def loop_complete(Range_Xb,C,Bo,alpha):
	#First step is to compute the values without gravity
	Fb,Fc = loop_simple([Range_Xb[i]-1 for i in range(len(Range_Xb))],C-1)
	#Remove the values that can be missing from unproper care of limit cases
	Fb = [Fb[i] for i in range(len(Fb)) if np.isnan(Fb[i])==False]
	Fc = [Fc[i] for i in range(len(Fc)) if np.isnan(Fc[i])==False]
	A = np.zeros(len(Fb))
	K = list()
	#Initialisation of the different force arrays
	Fcv,Fcb,Fct = np.zeros(len(Fb)),np.zeros(len(Fb)),np.zeros(len(Fb))
	Fbb,Fbt = np.zeros(len(Fb)),np.zeros(len(Fb))
	Fb = np.linspace(0.0,1.2,num = len(Fb))
	for i in range(len(Fb)):
		#Compute the three capillary forces moduli
		Fcv[i],Fcb[i],Fct[i] = CapillaryForce(Fb[i],Fc[i],C-1,Bo,alpha)
		#Compute the two extra neighbor forces moduli using eq. 29
		Fbb[i],Fbt[i] = Fb[i]-4.*Bo*np.pi*np.sin(alpha)/6., Fb[i]+4*Bo*np.pi*np.sin(alpha)/6
#Because the loop has many steps in it, it is always a good idea to check the result
#Here, the verification is performed by inputing all the forces into the expression of xbt and xbb, and comparing them to the inputed bubble length implicitly expressed in C (see eq. 26), see eqs. 30 and 31
		A[i] = LongitudinalLength(Fbt[i],Fbb[i],Fcv[i],Fct[i],Fcb[i])
	K.append([A,Fbb,Fbt,Fcv,Fcb,Fct])
	#This part will display an error message if final deformation values are not consistent with initial ones
	EcartLength=np.max([1.-1./C-0.5/C*xct(Fbt[i],Fbb[i],Fcv[i],Fct[i],Fcb[i]) -0.5/C*xcb(Fbt[i],Fbb[i],Fcv[i],Fct[i],Fcb[i]) for i in range(len(Fbb))])
	#if np.max([C-(1+(xct(Fbt[i],Fbb[i],Fcv[i],Fct[i],Fcb[i])+xcb(Fbt[i],Fbb[i],Fcv[i],Fct[i],Fcb[i]))/2))/C for i in range(len(Fbb))])>=1e-2:
	if EcartLength>=1e-2:
		print('Problem in xct and xcb')
		print('Imposed confinement is '+str(np.round(C,4)))
		print('Found confinement is '+str(1+(xct(Fbt[0],Fbb[0],Fcv[0],Fct[0],Fcb[0])+xcb(Fbt[0],Fbb[0],Fcv[0],Fct[0],Fcb[0]))/2))
	return K
#Values outputed : LB/2R0,Fbt,Fbb,Fcv,Fcb,Fct

#Functions relation ponctual deformations to the full forces, see eqs. 30-34
def xbt(fbt,fbb,fcv,fct,fcb):
	Delta = 8.*np.pi*np.exp(-5./6.)
	return log(fbt*Delta)*fbt/4/np.pi+fcv/4/np.pi+fcv/8/np.pi+fcb/8/np.pi-5/24/np.pi*fbb
	#return np.log(fbt*Delta)*fbt/4/np.pi+fcv/4/np.pi+fcv/8/np.pi+fcb/8/np.pi-5/24/np.pi*fbb

def xbb(fbt,fbb,fcv,fct,fcb):
	Delta = 8.*np.pi*np.exp(-5./6.)
	return log(fbb*Delta)*fbb/4/np.pi+fcv/4/np.pi+fcv/8/np.pi+fcb/8/np.pi-5/24/np.pi*fbt


def xct(fbt,fbb,fcv,fct,fcb):
	Delta = 8.*np.pi*np.exp(-5./6.)
	return fct*log(fct/Delta)/4/np.pi-5*fcb/24/np.pi+fcv/4/np.pi+(fbb+fbt)/8/np.pi

def xcb(fbt,fbb,fcv,fct,fcb):
	Delta = 8.*np.pi*np.exp(-5./6.)
	return fcb*log(fcb/Delta)/4/np.pi-5*fct/24/np.pi+fcv/4/np.pi+(fbb+fbt)/8/np.pi

def DisplayResult(Q,Bo,alpha):
#Values outputed : LB/2R0,Fbt,Fbb,Fcv,Fcb,Fct
    L,Fbt,Fbb,Fcv,Fcb,Fct=Q
    Fbt=[Fbt[i] for i in range(len(Fbt)) if np.isnan(L[i])==False and L[i]<=np.nanmin(L[0:i+1])]
    Fbb=[Fbb[i] for i in range(len(Fbb)) if np.isnan(L[i])==False and L[i]<=np.nanmin(L[0:i+1])]
    Fcv=[Fcv[i] for i in range(len(Fcv)) if np.isnan(L[i])==False and L[i]<=np.nanmin(L[0:i+1])]
    Fcb=[Fcb[i] for i in range(len(Fcb)) if np.isnan(L[i])==False and L[i]<=np.nanmin(L[0:i+1])]
    Fct=[Fct[i] for i in range(len(Fct)) if np.isnan(L[i])==False and L[i]<=np.nanmin(L[0:i+1])]
    L=[L[i] for i in range(len(L)) if np.isnan(L[i])==False and L[i]<=np.nanmin(L[0:i+1])]
#    Fbt=[Fbt[i] for i in range(len(Fbt)) if np.isnan(L[i])==False]
    fig=plt.figure(figsize=(8,12))
    plt.subplot(211)
    plt.title(r'$\alpha = $'+str(alpha)+", $B_0=$"+str(Bo))
    plt.xticks( [0.9,0.95,1.00,1.05],fontsize=14)
    plt.yticks([0,0.2,0.4,0.6,0.8],fontsize=14)
    plt.plot(L,[(Fbt[i]+Fbb[i])/2 for i in range(len(Fbt))],label="$FBm$")
    plt.plot(L,[(Fct[i]+Fcb[i])/2 for i in range(len(Fct))],label="$Fcm$")
    plt.legend(),plt.subplot(212)
    plt.xticks( [0.9,0.95,1.00,1.05],fontsize=14)
    plt.yticks([0,0.2,0.4,0.6,0.8],fontsize=14)
    plt.plot(L,Fbt,'--',label="$FBt$")
    plt.plot(L,Fbb,'--',label="$FBb$")
    plt.plot(L,Fcb,'--',label="FCb")
    plt.plot(L,Fcv,'--',label="FCv")
    plt.plot(L,Fct,'--',label="FCt")
    plt.legend(),plt.show()

#%%
print("Input parameter values\n  ")
C=input("Normalised bubble length LB/2R0 :")
Bo=input("Bond number :")
alpha=input("Angle with horizontal axis :")


print("====================\n L_B/2R0="+str(C))
Range_Xb = np.linspace(0.7,1.2,num = 2000)

Fb,Fc = loop_simple([Range_Xb[i]-1 for i in range(len(Range_Xb))],C-1)
Range_Xb = [Range_Xb[i] for i in range(len(Range_Xb)) if np.isnan(Fb[i])==False]
Q0 = loop_complete(Range_Xb,C,Bo,alpha)
#Q25 = loop_complete(Range_Xb,C,0.08,alpha)
DisplayResult(Q0[0],Bo,"$\pi/2$")


