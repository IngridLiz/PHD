# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:44:00 2025

@author: ingrid.lizcano
"""

######
#Code to replicate Voorsluijs et al. ODE
#19 DE
#with caer and Jpdh (Bernt)
#res = solve_ivp(metcal_ode, [0, sec], y1, method='DOP853',t_eval=ts, atol=1.e-11, rtol=1.e-11)


import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import OdeSolver
from matplotlib import pyplot as plt

#time
sec=10000
step=sec*100
ts = np.linspace(0, sec, step)

# Conversions (nmol/(ml*min) = uM/min)
uMmM = 1000.            # (uM/mM)
nMmM = 1.0e6
#Cte values 
ip3 = 0.5
pyrm= 0.01
##############################################################################
## Parameters

# Jpyrexch
V_max_pyrex= 128.0  #mMs-1  Limiting rate of pyruvate exchanger                     (Berndt et al., 2015)
K_mpyrc= 0.15       #mM     Michaelis constant of PyrEx for cytosolic pyruvate      (Berndt et al., 2015)
K_mpyrm= 0.15       #mM     Michaelis constant of PyrEx for mitochondrial pyruvate  (Berndt et al., 2015)
    # Concentrations needed:
    #hc= 6.31e-5       
    #hm = 1.0e-5        

# Jpdh
#"""
#Berndt2012
#V_max_pdh= 13.1     #mM     Limiting rate of Pyruvate Dehydrogenase                 (Berndt et al., 2015)
V_max_pdh= 0.1      #mMs-1
A_max_cam = 1.7     #       Modulator factor for Ca2+ regulation                    (Berndt et al., 2015 and 2012)
K_a_cam = 1.0e-3    #uM     Activation constant of pdh for mitochondrial Ca2+       (Berndt et al., 2015 and 2012)
K_m_pyr = 0.090     #mM     Michaelis constant of pdh for mitochondrial pyruvate    (Berndt et al., 2012)
K_m_NAD = 0.036     #mM     Michaelis constant of pdh for NAD                       (Berndt et al., 2012)
K_m_coam = 0.0047   #mM     Michaelis constant of pdh for coam                      (Berndt et al., 2015 and 2012)
#"""
"""
#Berndt2015
V_max_pdh= 13.1     #mM     Limiting rate of Pyruvate Dehydrogenase                 (Berndt et al., 2015)
A_max_cam = 1.7     #       Modulator factor for Ca2+ regulation                    (Berndt et al., 2015 and 2012)
K_a_cam = 1.0e-3    #mM     Activation constant of pdh for mitochondrial Ca2+       (Berndt et al., 2015 and 2012)
K_m_pyr = 0.068     #mM     Michaelis constant of pdh for mitochondrial pyruvate    (Berndt et al., 2012)
K_m_NAD = 0.041     #mM     Michaelis constant of pdh for NAD                       (Berndt et al., 2012)
K_m_coam = 0.0047   #mM     Michaelis constant of pdh for coam                      (Berndt et al., 2015 and 2012)
"""
"""
#Tseng
V_max_pdh= 0.3       #mM     Limiting rate of Pyruvate Dehydrogenase                   Wen-Wei Tseng
K_a_cam= 50/nMmM
K_m_pyr= 0.0475      #mM
K_m_NAD= 81/uMmM
u1=1.5
u2=1.1
"""
"""
#Fridlyand
V_max_pdh= 0.3       #mM     Limiting rate of Pyruvate Dehydrogenase                   Wen-Wei Tseng
K_a_cam= 50/nMmM     #mM
K_m_pyr= 0.0475      #mM
K_m_NAD= 81/uMmM   
u1=1.5
u2=1.1
"""
"""
#Bertram

p1ber=400.0
p2ber=1.0
p3ber=0.01     	                    #um
FBP=5.0                             #um
kgpdh=5.0e-4                        #um/ms
#Jgpdh= kgpdh*np.sqrt(FBP/1)         #um/ms = mM/s
Jgpdh= 450/uMmM             #mM/s     Velocity of glycolysis (empirical)   
"""
"""
#Wacquier
FBP=15.0                             #um
kgpdh=5.0e-4                        #um/ms
Jgpdh= kgpdh*np.sqrt(FBP/1)         #um/ms = mM/s
k_gly= 400.   /uMmM	          #mM/s     Velocity of glycolysis (empirical)                 
q1=1.0              #       Michaelis-Menten-like constant for NAD+ consumption by the Krebs cycle
q2=0.1/uMmM	        #mM     value for activation the Krebs cycle by Ca2+
"""

#"""
#MK
Fpdh=0.4
Jgpdh=1.0 /uMmM	
vpdh= 77.0 /uMmM	
delta= 0.1
K_a_cam=0.1 /uMmM	
K_a_nad=1.0
u1= 1.5
u2=1.0

#"""

# Jcs 
    # -Citrate synthase (CS)- (Voorslujis et al., 2024)
Vmaxcs = 104.0          #mMs-1  Limiting rate of CS                                 (Voorslujis et al., 2024)
kmaccoa = 1.2614e-2     #mM     Michaelis constant of CS for AcCoa                  (Cortassa et al.,2003) Dudycha 2000: 1.26e-2 mM; BRENDA: 5-10 uM, Shepherd-Garland 1969: 16 uM)
kiaccoa = 3.7068e-2     #mM     Inhibition constant of CS for AcCoA                 (Cortassa et al.,2003 or Dudycha 2000) ?
ksaccoa = 8.0749e-2     #mM     Binding constant of citrate synthase for AcCoA      (Dudycha 2000)
kmoaa = 5.0e-3          #mM     Michaelis constant of CS for oxaloacetate           (Bernart et al., 2015; Matsuoka and Srere; Kurz et al.)
#other values for kmoaa = Cortassa 2003, Dudycha 2000: 6.2981e-4, Shepherd-Garland: 0.002, Kurz...Srere: 0.0059 mM, Matsuoka & Srere 1973+Berndt 2015 = 0.0050 mM

#"""
# -Citrate synthase- Farina PhD thesis 2022
#Vmaxcs = 52         #mMs-1  Limiting rate of CS                             

#"""
#Jaco
    # -Aconitase (ACO)-
keaco = 0.067           #       Equilibrium constant of aconitase                   (Berndt et al.,2015; Garret & Grisham) equilibrator pH=8, I=0.12M, pMg=3.4, 
kfaco = 12.5            #s-1    Forward rate constant of aconitase (1/s)            (Cortassa et al., 2003) 

#Jidh
    # -Isocitrate dehydrogenase (IDH)-
Vmaxidh = 1.7767    #mMs-1      Limiting rate of IDH                                (Cortassa et al., 2003) push conditions
kmisoc = 1.52       #mM         Michaelis constant of IDH for isocitrate            (Cortassa et al., 2003)
kmnad = 0.923       #mM         Michaelis constant of IDH for NAD                   (Cortassa et al., 2003)
kinadh = 0.19       #mM         Inhibition constant of IDH for NADHm                (Cortassa et al., 2003)
kaadp = 6.2e-2      #mM         Activation constant of IDH for ADPm                 (Dudycha 2000 and Cortassa et al., 2003)
kaca = 1.41         #uM         Activation constant of IDH for mitochondrial Ca2+   (Cortassa et al., 2003)
kh1 = 8.1e-5        #mM         First ionization constant of IDH                    (Dudycha 2000 and Cortassa et al., 2003)
kh2 = 5.98e-5       #mM         Second ionisation constant of IDH                   (Dudycha 2000 and Cortassa et al., 2003)
ni = 2.0            #           Hill coefficient of IDH for isocitrate              (Wei et al., 2011 and Cortassa et al., 2011) Cooperativity of Isoc in IDH (lacking in Cortassa 2003)
    # Concentrations needed:      
    #hm = 1.0e-5  

#Jkgdh
    # -Alpha-ketoglutarate dehydrogenase (alfa-kg)-
Vmaxkgdh = 2.5          #mMs-1      Limiting rate of aKG dehydrogenase                      (Cortassa et al., 2003)
kmakg = 1.94            #mM         Michaelis constant of KGDH for aKG                      (Cortassa et al., 2003)
kmnadkgdh = 0.0387      #mM         Michaelis constant of KGDH for NAD                      (Voorslujis et al., 2024)
kdca = 1.27             #uM         Dissociation constant of KGDH for mitochondrial Ca2+    (Cortassa et al., 2003)
kdmg = 0.0308           #mM         Dissociation constant of KGDH for Mg2+                  (Cortassa et al., 2003)
nakg = 1.2              #           Hill coefficient of KGDH for aKG                        (Cortassa et al., 2003) 
    # Concentrations needed:      
    # mg = 0.4     

#Jsl
    #  -Succinyl-CoA Lyaseor Succinyl-CoA Synthetase (SL)- 
kesl = 0.724            #           Equilibrium constant of succinyl coa lyase              (Flamholz et al., 2012)  equilibrator pH=8, I=0.12M, pMg=3.4 (reac with ADP/ATP: 0.724, with GDP/GTP: 2.152)
kfsl = 0.127            #mM-1s-1    Forward rate constant of succinyl coa lyase             (Cortassa et al., 2003)
    # Concentrations needed:      
    # pim =20 

#Jsdh
    #   -Succinate dehydrogenase (SDH)-
Vmaxsdh = 0.5           #mMs-1      Limiting rate of succinate dehydrogenase                (Cortassa et al., 2003)
kmsuc = 3.0e-2          #mM         Michaelis constant of SDH for succinate                 (Cortassa et al., 2003)
kioaasdh = 0.15         #mM         Inhibition constant of SDH for OAA                      (Cortassa et al., 2003)
kifum = 1.3             #mM         Inhibition constant of SDH for fumarate                 (Cortassa et al., 2003)

#Jfh
    # -Fumarate hydratase (FH)-
kefh = 3.942            #           Equilibrium constant of FH                              (Flamholz et al., 2012)  equilibrator pH=8, I=0.12M, pMg=3.4
kffh = 8.3              #s-1        Forward rate constant of FH                             (Cortassa et al., 2003)

#Jmdh
    # -Malate dehydrogenase (MDH)-
Vmdh = 128.0            # mMs-1     Limiting rate of MDH                                    (Voorslujis et al., 2024)
Keqmdh = 2.756e-5       #           Equilibrium constant of MDH                             (Flamholz et al., 2012) equilibrator pH=8, I=0.12M, pMg=3.4 
#other values of Keqmdh: 
    #Berndt 2015 : e4 
kmmal = 0.145           #mM         Michaelis constant of MDH for malate                    (Berndt et al.,2012)
#other values of kmmal: 
    #Berndt 2015 : 0.77 
kmnadmdh = 0.06         #mM         Michaelis constant of MDH for NAD                       (Berndt et al.,2012)
#other values of kmnadmdh: 
    #Berndt 2015 : 0.05
kmoaamdh = 0.017        #mM         Inhibition constant of MDH for OAA                      (Berndt et al.,2012)
#other values of kmoaamdh: 
    #Berndt 2015 : 0.04
kmnadhmdh = 0.044       #mM         Michaelis constant of MDH for NADH                      (Berndt et al.,2012)
#other values of kmnadhmdh: 
    #Berndt 2015 : 0.05


#Jf1
    # --- ATPase --- #
psiB = 50.0              #mM         Total phase boundary potentials (mV)                        (Magnus and Keizer 1997) Tables 2 and 3
p1 = 1.346e-8            #                                                                       (Magnus and Keizer 1997) Table 3
p2 = 7.739e-7            #                                                                       (Magnus and Keizer 1997) Table 3  
p3 = 6.65e-15            #                                                                       (Magnus and Keizer 1997) Table 3  
pa = 1.656e-5            #1/s                                                                    (Magnus and Keizer 1997) Table 3  
pb = 3.373e-7            #1/s                                                                    (Magnus and Keizer 1997) Table 3  
pc1 = 9.651e-14          #1/s                                                                    (Magnus and Keizer 1997) Table 3  
pc2 = 4.845e-19          #1/s                                                                    (Magnus and Keizer 1997) Table 3  
kf1 = 1.71e6             #mM          Equilibrium constant of ATP hydrolysis                     (Cortassa 2003, 2006, 2011)(Magnus and Keizer 1997) Tables 3  
    
    # Concentrations needed:      
        # rhof1 = 0.23
        # pim =20
        # DpH = -0.8
 
#Jhl    
    #  --- M. Proton leak ---#
gH = 1.e-5                  # mM/(mV * s)    Ionic conductance of the mitochondrial inner membrane              (Cortassa et al., 2003)
    # Concentrations needed:      
        # DpH = -0.8

#Jhyd 
    # -- Hydrolisis -- # 
khyd = 9.e-2                #mMs-1          Hydrolysis rate of ATPc due to cellular activity                    (Voorslujis et al., 2024) Basal hydrolysis rate of ATPc (1/s)
katpcH = 1.00               #mM             Michaelis constant for ATPc hydrolysis due to cellular activity     (Wacquier et al., 2016)

#Jo
     # -- Oxidation/ Respiration -- #  
r1 = 2.077e-18              #                                                                                   (Magnus and Keizer 1997) Table 2                  
r2 = 1.728e-9               #                                                                                   (Magnus and Keizer 1997) Table 2
r3 = 1.059e-26              #                                                                                   (Magnus and Keizer 1997) Table 2
ra = 6.394e-10              #1/s                                                                                (Magnus and Keizer 1997) Table 2
rb = 1.762e-13              #1/s                                                                                (Magnus and Keizer 1997) Table 2
rc1 = 2.656e-19             #1/s                                                                                (Magnus and Keizer 1997) Table 2
rc2 = 8.632e-27             #1/s                                                                                (Magnus and Keizer 1997) Table 2
g = 0.85                    #                   Fitting factor for voltage                                      (Magnus and Keizer 1997) Table 2
kres = 1.35e18              #                                                                                   (Magnus and Keizer 1997) Table 2
    # Concentrations needed:      
        # rhores = 1.00

#Jant
    # -- Adenine nucleotide transporter (Antiporter) -- #     
vant = 4.0                  #mMs-1           Limiting rate of adenine nucleotide translocator                   (Voorslujis et al., 2024)
#other values of vant:
    #Fall-Keizer 2001:900 nmol/(mg * min) => 900 * 1000 / 60 = 15 mM/s
alphac = 0.11111            #                = 0.05 / 0.45                                                      (Cortassa et al., 2003)
alpham = 0.1388889          #                = 0.05 / (0.45 * 0.8)                                              (Cortassa et al., 2003)
fant = 0.5                  #                fraction effective psi                                             (Magnus and Keizer 1997) Table 5 
#other values of fant:
    #Cortassa 2003:

## ER
#Jerout
        # --- ER LEAK and IP3Rs ---#
vip3 = 30.0                 #1/s     15. max release rate of Ca2+ through IP3R                                  (Voorslujis et al., 2024)
vleak = 0.15                #1/s     0.15 rate of Ca2+ release by leakage from ER 
kcai1 = 1.4                 #uM      Inhibition constant 1 of IP3R for Ca2+                                     Moeien thesis Table 3.1 : 1.3)
kcai2 = kcai1               #uM      Inhibition constant 1 of IP3R for Ca2+                                     Moeien thesis Table 3.1 : 1.3)
kcaa = 0.70                 #uM      Activation constant of IP3R for Ca2+                                       Moien 2017 : 0.9
kip3 = 1.00                 #uM      Activation constant of IP3R for IP3                                        (Wacquier et al., 2016)

#Jncx
        # --- NCX ---#
vncx = 2.0e-3              #mMs-1     Limiting rate of Na+/Ca2+ exchanger                                  (Voorslujis et al., 2024)
psiStar = 91.0                   # psi offset for Ca2+ transport (mV) -- MK 1997 Table 5
kna = 9.4                       # Km (Na+) for NCX (mM) -- MK 1997 Table 5
kca = 0.375                     # Km (Ca2+) for NCX (uM) -- Bertram 2006, Cortassa 2003
nac = 10.0                  # cytosolic Na+ concentration (mM)
nam = 5.0                        # mitochondrial Na+ concentration (mM)
n = 3.0                     # nb of Na+ binding to NCX (electroneutral/electrogenic: n=2/3) Limiting rate of Na+/Ca2+ exchanger-- MK 1997 Table 5
b = 0.5                         # NCX dependence on psi (electroneutral/electrogenic: b=0/0.5) -- MK 1997 Table 5

#Jserva
    # --- SERCA ---#
#vserca = 0.12   #entre 0.08 y 0.07                 # max SERCA rate (mM/s) -- Moeien 2017 0.455 but multiplied by Ver/Vc~1/3 0.15
#kserca = 0.35   # under 0.34 and over 0.72 GLY stops being steady               # Km of SERCA pumps for Ca2+ (uM) -- Wacquier 2016
#katpc = 0.05    #over 0.792               # Km of Serca for ATPc (mM) -- Wacquier 2016 (Lytton): 5.e-5 (mM), Moien 2017: 0.06,

vserca = 0.12                  # max SERCA rate (mM/s) -- Moeien 2017 0.455 but multiplied by Ver/Vc~1/3 0.15
kserca = 0.35               # Km of SERCA pumps for Ca2+ (uM) -- Wacquier 2016
katpc = 0.05

                               # Scofano 1979: 0.05 (mM) RefV:0.01
#Juni
    # --- MCU ---#
vuni = 0.300                    # 0.625 ; max uniporter rate at psi=psiStar (mM/s); Cortassa: 0.625 mM/s
psiStar = 91.                   # psi offset for Ca2+ transport (mV) -- MK 1997 Table 5
L = 110.                        # Keq for uniporter conformations -- MK 1998a Table 1
ma = 2.8                        # uniporter activation cooperativity -- MK 1997 Table 5
ktrans = 19.                    # Kd for uniporter translocated Ca2+ (uM) -- MK 1998a Table 1
kact = 0.38                     # Kd for uniporter activating Ca2+ (uM) -- MK 1997 Table 5
fant = 0.5                         # fraction effective psi -- MK 1997 Table 5, Cortassa 2003



##############################################################################
    # Concentrations:
hc= 6.31e-5         #mM     Cytosolic proton concentration                                  (Buckler et al., 1990 and Casey et al., 2010)
hm = 1.0e-5         #mM     Concentration of proton in the mitochondrial matrix             (Buckler et al., 1990 and Casey et al., 2010)
mg = 0.4            #mM     Concentration of mitochondrial Mg2+                             (Cortassa et al., 2003)    
pim = 20.0          #mM     Concentration of Inorganic phosphate in mitochondrial matrix    (Magnus and Keizer 1997)
coa = 0.02          #mM     Concentration of coenzyme A                                     (Cortassa et al., 2003) 
rhof1 = 0.23        #mM     Concentration of ATPase pumps                                   (Voorslujis et al., 2024)
DpH = -0.8          #       Ph difference beteen cytosol and mito matrix= pHc - pHm         (Casey et al., 2010)
rhores = 1.00       #mM     Respiration-driven H+ pump concentration                        (Magnus and Keizer 1997) Tables 2
#other values rhores:
    #Cortassa 2003: 7.2e-4 ?

    # Cell parameters
fc = 0.01               #        fraction of free Ca2+ in cytosol -- MK 1998a, Fall-Keizer 2001
fe = 0.01               # fraction of free Ca2+ in ER -- Fall-Keizer 2001
fm = 0.0003             #        fraction of free Ca2+ in mitochondria -- Fall-Keizer 2001
alpha = 0.10            # Ve / Vc -- volumic ratio between ER and cyt
delta = 0.15            # Vm / Vc -- volumic ration between mito and cyt
cmito = 1.812e-3        # mitochondrial membrane capacitance (mM/mV)
#deltaer= 0.09           # fraction of free ca in er * cytosolic to er volum ratio                # moein these
deltaer=0.1             # valerie
ctot = 1500.0                    # = cac + alpha*caer/fe + delta*cam/fm (uM)
coef1 = fe * ctot / alpha               # coefficients allowing for calculation of caer
coef2 = fe / (alpha * fc)
coef3 = delta * fe / (alpha * fm)

RTF = 26.712338         #mV       = R*T/F, R is the gas cte (8.314 J/(mol*K)), T is the abs temp (310K), F is Faraday’s cte (96485 C/mol)      (Magnus and Keizer 1997)



###############################################################################

# Conservation relations
amtot = 15.                     # = atpm + adpm (mM)
atot = 3.                       # = atpc + adpc (mM)
ckint = 1.                      # = cit + isoc + akg + scoa + suc + fum + mal + oaa (mM) -- Cortassa 2003
nadtot = 0.8                    # = nadm + nadhm (mM)
#nadtot = 1                   # = nadm + nadhm (mM)
ctot = 1500.                    # = cac + alpha*caer/fe + delta*cam/fm (uM)


##############################################################################
## Fluxes
flux=[]
Jaco_v=[]
Jant_v=[]
Jcs_v=[] 
Jerout_v=[] 
Jf1_v=[]
Jfh_v=[]
Jhl_v=[]
Jidh_v=[]
Jkgdh_v=[]
Jmdh_v=[]
Jncx_v=[]
Jo_v=[]
Jsdh_v=[]
Jserca_v=[]
Jsl_v=[]
Juni_v=[]
Jhyd_v=[]
Jpdh_v=[]

def fluxes(v): 
    [adpc, adpm, akg, atpc, atpm, cac, cam, cit, fum, isoc, mal, nadm, nadhm, oaa, psi, scoa, suc, caer, accoa ]= v
    
    #Jpyrexch = V_max_pyrex * (pyrc * hc - pyrm * hm) / (((1 + pyrc)/K_mpyrc) * ((1 + pyrm)/K_mpyrm))                                                    # (Berndt et al., 2015)
    
    
    
    Jpdh = V_max_pdh * (1 + (A_max_cam * cam / (cam + K_a_cam))) * (pyrm / (pyrm + K_m_pyr)) * (nadm / (nadm + K_m_NAD)) * (coa / (coa + K_m_coam))     # (Berndt et al., 2012)
    
    #Jpdh = (p1ber*nadm/(p2ber*nadm+nadhm))*(cam/(p3ber+cam))*(Jgpdh+0.0005)                                                                                                  # (Bertram et al., 2006)
    
    #calpdh=(1.0+(cam/K_a_cam))**2                                                                                                                        # (Tseng et al., 2024)
    #Jpdh = V_max_pdh* (pyrm / (pyrm + K_m_pyr))*(nadm / (nadm + K_m_NAD))* calpdh/((calpdh*(1.0+u1))+u2*u1)                                              # (Tseng et al., 2024)
    
    #Jpdh = V_max_pdh* (pyrm / (pyrm + K_m_pyr))*((nadm/nadhm) / ((nadm/nadhm) + K_m_NAD))* (1/((1+u2)*(1+u1*(1+(cam/K_a_cam))**(-2))) )                  # (Fridlyand et al., 2010)

    #Jpdh = k_gly*(1/(q1+(nadhm/nadm)))*(cam*(q2+cam))
    
    #calpdh=(1.0+(cam/K_a_cam))**2 
    #Fpdh=1.0/(1.0+u2*(1.0+(u1/(calpdh*((nadhm/nadm)+K_a_nad)))))
    
    #calpdh=(1.0+(cam/K_a_cam))**2 
    #Jpdh=1.0/(1.0+u2*(1.0+(u1/(calpdh))))*130.0
    
    #Jpdh = 0.05                                                                                                                   # mM     (spikes )
    
    
    ## TCA
    # Citrate synthase (CS)
    Jcs = Vmaxcs * accoa / ( accoa + kmaccoa + ((kmoaa * accoa / oaa) * (1. + accoa / kiaccoa)) + (ksaccoa * kmoaa) / oaa)                              # (Dudycha 2000)
    #Jcs = 0.02
    
    # Aconitase (ACO)
    Jaco = kfaco * (cit - isoc / keaco)                                                                                                                 # (Cortassa et al., 2003)  
    
    #Isocitrate dehydrogenase (IDH)
    Jidh = Vmaxidh / (1. + (hm / kh1) + (kh2 / hm) + (((kmisoc / isoc) ** ni) / ((1. + adpm / kaadp) * (1. + cam / kaca))) + (
            kmnad / nadm) * (1. + (nadhm / kinadh)) + ((((kmisoc / isoc) ** ni )* (kmnad / nadm) * (
                              1. + (nadhm / kinadh))) / ((1. + (adpm / kaadp)) * (1. + (cam / kaca)))))                                                 # (Cortassa et al., 2003)  

    # Alpha-ketoglutarate dehydrogenase (akg)
    Jkgdh = Vmaxkgdh / (1. + (((kmakg / akg)* ((kmnadkgdh / nadm) ** nakg))/((1. + (mg / kdmg)) * (1. + (cam / kdca))))) #revie exponenets              # (Dudycha 2000)
                              
    # Succinyl Coa Lyase (SL)
    Jsl = kfsl * (scoa * adpm * pim  - suc * atpm * coa / kesl)                                                                                         # (Wei et al., 2011) Cortassa does not has P
    
    # Succinate dehydrogenase (SDH)
    Jsdh = Vmaxsdh / (1. + ((kmsuc / suc) * (1. + oaa / kioaasdh) * (1. + fum / kifum)))                                                                # (Cortassa et al., 2003) 
    
    # Fumarate hydratase (FH)
    Jfh = kffh * (fum - mal / kefh)                                                                                                                     # (Cortassa et al., 2003) 
    
    
    # Malate dehydrogenase
    #"""
    Jmdh = Vmdh * (mal * nadm - (oaa * nadhm / Keqmdh)) / (
                (1. + mal / kmmal) * (1. + nadm / kmnadmdh) + ((1. + oaa / kmoaamdh) * (
                    1. + nadhm / kmnadhmdh)) - 1.)  
    #"""                                                                                                    # (Berndt et al., 2015)
    #Jmdh=0.02
    ##  Cell fluxes 
    
    # Phosphorylation of mitochondrial ADP via F1F0 ATPase
    VDf1 = np.exp((3.0 * psi)/ RTF) 	     #var
    VDf1B = np.exp(3.0 * psiB / RTF)        #cte
    Pa1 = (pa*(10**(3.0*DpH)))              #cte
    Pac1 = (Pa1 + (pc1 * VDf1B))            #cte
    Af1 = kf1 * atpm / (adpm * (pim/uMmM))                #pim in M ? review units
    
    Jf1 = - rhof1 * ((Pac1 * Af1 )- (pa * VDf1) + (pc2 * Af1 * VDf1) )/ (
        ((1.0 + (p1 * Af1)) * VDf1B )+ ((p2 + (p3 * Af1)) * VDf1))                                                                                      #(Magnus and Keizer 1997)                        
    
    
    # Proton leak from cytosol to mitochondria
    deltaMuH = psi - (2.303 * RTF * DpH)
    
    Jhl = gH * deltaMuH                                                                                                                                 #(Magnus and Keizer 1997)
    
    
    # Cytocolic ATP hydrolisis - cell activity
    Jhyd= khyd * atpc / (katpcH + atpc)                                                                                                                 #(Wacquier et al., 2016)
    
    
    # Oxidation of NADH by respiration
    VDres = np.exp(6.0 * g * psi / RTF)      #var
    VDresB = np.exp(6.0 * psiB / RTF)         #cte
    
    Ares = kres * (nadhm** 0.5  / nadm** 0.5 )    #cte
    Ra1 = ra * (10 ** (6.0*DpH))              #cte
    Rac1 = (Ra1 + rc1 * VDresB)               #cte
    
    Jo = 0.5 * rhores * ((Rac1 * Ares) - (ra * VDres) + (rc2 * Ares * VDres)) / (
        ((1.0 + r1 * Ares) * VDresB) + ((r2 + (r3 * Ares)) * VDres) )                                                                                   #(Magnus and Keizer 1997)
    
    
    # Adenine nucleotide transporter
    Jant = vant * (1.0 - alphac * atpc * adpm * np.exp(-psi / RTF) / (alpham * adpc * atpm)) / (
                (1.0 + alphac * atpc * np.exp(-fant * psi / RTF) / adpc) * (1. + adpm / (alpham * atpm)))                                               #(Magnus and Keizer 1997)
    
    ## ER
    # Ca2+ release from ER via IP3Rs + passive leak
    # Calcium in the ER is calculated based on conservation of the total number of moles of calcium in the cell
    # caer = coef1 - cac * coef2 - cam * coef3  # uM
    
    Jerout= ((vip3 * ((ip3 ** 2.0) / ((ip3 ** 2.0) + (kip3 ** 2.))) * ((cac ** 2.0) / ((cac ** 2.0) + (kcaa ** 2.0))) * 
                ((kcai1 ** 4.0) / ((kcai2 ** 4.0) + (cac ** 4.0)))) + vleak) * (caer - cac) / uMmM                                                      #(Komin et al., 2015)                                    
    
    # Ca2+ release from mitochondria via NCX 
    Jncx = vncx * (cam / cac) * np.exp(b * (psi - psiStar) / RTF) / ((1. + kna / nac) ** n * (1. + kca / cam))                                          # (Cortassa et al., 2003) 
    
    # SERCA pumps
    Jserca = vserca * (cac ** 2. / (kserca ** 2. + cac ** 2.)) * (atpc / (atpc + katpc))                                                                #(Wacquier et al., 2016)
    
    # Ca2+ uptake to mitochondria via MCU
    VDuni = 2.0 * (psi - psiStar) / RTF
    trc = cac / ktrans
    mwc = trc * ((1. + trc) ** 3.0) / (((1.0 + trc) ** 4.0) + (L / ((1.0 + (cac / kact)) ** ma)))
    
    Juni = (vuni * VDuni * mwc )/ (1.0 - np.exp(-VDuni))                                                                                                   #(Magnus and Keizer 1997)
    
    
    flux = [Jpdh_v.append(Jpdh), Jcs_v.append(Jcs), Jaco_v.append(Jaco),  Jidh_v.append(Jidh), Jkgdh_v.append(Jkgdh) , 
            Jsl_v.append(Jsl), Jsdh_v.append(Jsdh), Jfh_v.append(Jfh), Jmdh_v.append(Jmdh), Jerout_v.append(Jerout)
            ,Jserca_v.append(Jserca), Juni_v.append(Juni), Jncx_v.append(Jncx), Jf1_v.append(Jf1), Jhl_v.append(Jhl) 
            ,Jant_v.append(Jant), Jo_v.append(Jo), Jhyd_v.append(Jhyd)]
    
    
    return   Jaco, Jant, Jcs, Jerout, Jf1, Jfh, Jhl, Jidh, Jkgdh, Jmdh, Jncx, Jo, Jsdh, Jserca, Jsl, Juni, Jhyd, Jpdh, flux ;



def metcal_ode(s, v):
    # Evolution equations for all the species of interest
    # Returns ODEs evaluated at time s (in mV/s, uM/s or mM/s))
    # Calls fluxes(v) defined above to calculate the fluxes
    # v is a vector containing the concentrations of the species of interest

    # f_glc = 0.048
    
    # Compute fluxes
    Jaco, Jant, Jcs, Jerout, Jf1, Jfh, Jhl, Jidh, Jkgdh, Jmdh, Jncx, Jo, Jsdh, Jserca, Jsl, Juni, Jhyd, Jpdh, flux  = fluxes(v)
    
    dvds = np.zeros((len(v), ))
                                            
    #f_pyr= delta * Jpyrexch
    dvds[0] = + (-delta * Jant + Jhyd + 0.5 * Jserca)  #ADP_c mM/s
    dvds[1] = Jant - Jf1 - Jsl  # adpm mM/s
    dvds[2] = Jidh - Jkgdh #+ Jaat  # akg mM/s
    dvds[3] = - (-delta * Jant + Jhyd + 0.5 * Jserca)  #ATP_c mM/s
    dvds[4] = -(Jant - Jf1 - Jsl)  # atpm mM/s
    dvds[5] = uMmM * fc * (-Jserca + Jerout + delta * (Jncx - Juni))  # cac uM/s
    dvds[6] = uMmM * fm * (Juni - Jncx)  # cam uM/s
    dvds[7] = Jcs - Jaco  # cit mM/s
    dvds[8] = Jsdh - Jfh  # fum mM/s
    dvds[9] = Jaco - Jidh  # isoc mM/s
    dvds[10] =  Jfh - Jmdh  # mal mM/s
    dvds[11] = -(-Jo + Jidh + Jkgdh + Jmdh +Jpdh )  # nadm mM/s
    dvds[12] = -Jo + Jidh + Jkgdh + Jmdh + Jpdh  # nadhm mM/s
    dvds[13] = Jmdh - Jcs #- Jaat  # oaa mM/s
    dvds[14] = (10. * Jo - 3. * Jf1 - Jant - Jhl - Jncx - 2. * Juni) / cmito  # psi mV/s
    dvds[15] = Jkgdh - Jsl  # scoa mM/s
    dvds[16] = Jsl - Jsdh  # suc mM/s
    dvds[17] = uMmM * deltaer *(Jserca - Jerout) # Caer uM/s
    dvds[18] = Jpdh -Jcs #AcCoA
 
    
    return dvds



# Read initial conditions
y0 = np.load("initial_conditions_15july.npy", allow_pickle=True)
ic_glc0, ic_atpc0, ic_adpc0, ic_glyc0, ic_pyrc0, ic_lacc0, ic_pyrm0, ic_accoa0, ic_adpm0, ic_akg0, ic_atpm0, ic_cac0, ic_cam0, ic_cit0, ic_fum0, ic_isoc0, ic_mal0, ic_nadhm0, ic_nadm0, ic_oaa0, ic_psi0, ic_scoa0, ic_suc0 = y0

#y1= [ic_pyrc0,  ic_pyrm0, ic_cam0, ic_nadm0, ic_accoa0, ic_oaa0, ic_cit0, ic_isoc0, ic_adpm0, ic_nadhm0, ic_akg0, ic_scoa0, ic_suc0, ic_atpm0, ic_fum0, ic_mal0, ic_psi0, ic_atpc0, ic_adpc0, ic_cac0]

adpc0 = atot*0.20               # mM
adpm0 = amtot*0.50              # mM
akg0 = ckint*0.01               # mM
asp0 = 0.                       # mM
atpc0 = atot - adpc0            # mM
atpm0 = amtot - adpm0           # mM
cac0 = 0.20                    # uM
cam0 = 0.10                    # uM
fum0 = ckint*0.01               # mM
isoc0 = ckint*0.01              # mM
mal0 = ckint*0.01               # mM
nadhm0 = 0.125*nadtot           # mM
nadm0 = nadtot - nadhm0         # mM
oaa0 = ckint*1.e-3              # mM
psi0 = 160.                     # mV
scoa0 = ckint*0.01              # mM
suc0 = ckint*0.01               # mM
cit0 = ckint - isoc0 - akg0 - scoa0 - suc0 - fum0 - mal0 - oaa0 # mM

#new
#cac0 = 1.25                    # uM
#cam0 = 0.30                    # uM
caer0= coef1 - cac0 * coef2 - cam0 * coef3 # uM
accoa0= 0.01

y1= [adpc0, adpm0, akg0, atpc0, atpm0, cac0, cam0, cit0, fum0, isoc0, mal0, nadm0, nadhm0, oaa0, psi0, scoa0, suc0, caer0, accoa0 ]
     
# Solve the ODE
#res = solve_ivp(metcal_ode, [0, sec], y1, method='DOP853', atol=1.e-5, rtol=1.e-5)
#res = solve_ivp(metcal_ode, [0, sec], y1, method='LSODA', atol=1.e-11, rtol=1.e-11)
res = solve_ivp(metcal_ode, [0, sec], y1, method='DOP853',t_eval=ts, atol=1.e-11, rtol=1.e-11)
#res = OdeSolver(metcal_ode, 0, y1, sec, False)
#res = solve_ivp(metcal_ode, [0, sec], y1, method='RK23', t_eval= ts)

#res = solve_ivp(metcal_ode, [0, sec], y1, method='LSODA')
#Correction of variables from sofia
variables = {
    "ADP_c mM": 0, "ADP_m mM": 1, "akg mM": 2, "ATP_c mM": 3, "ATP_m mM": 4, "Ca_c uM": 5,
    "Ca_m uM": 6, "Cit mM": 7,"Fum mM": 8, "Isoc mM": 9,
    "Mal mM" :10,"NAD_m mM": 11,"NADH_m mM": 12, "Oaa mM": 13, "Psi mV": 14,
    "Scoa mM": 15, "Suc mM": 16, "Ca_er uM": 17, "AcCoa mM": 18
}
# Example: Choose which variables to plot using their dictionary keys
variables_to_plot = list(variables.keys())  # Plotting all variables

# Function to extract unit from the variable name
def extract_unit(variable_name):
    parts = variable_name.split()
    return parts[-1] if len(parts) > 1 else 'a.u.'  # 'a.u.' for arbitrary unit if no unit is provided

# Determine the number of rows and columns for subplots
n_cols = 4
n_rows = (len(variables_to_plot) + n_cols - 1) // n_cols  # Calculate required rows

###
### Plot all concentrations
###
#"""
# Create subplots

timev=res.t
rang=sec-2000
#uplim= np.where(timev == sec)[0][0]
#downlim= np.where(timev <= rang)[0][-1]
#[downlim:uplim]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 1.3 * n_rows), sharex=True)
# Flatten axes for easier iteration

# Flatten axes for easier iteration
axes = axes.flatten()
# Plot each chosen variable
for i, var in enumerate(variables_to_plot):
    var_index = variables[var]
    axes[i].plot(res.t, res.y[var_index], label=f'{var}')
    #axes[i].set_ylabel(f'{var}')
    axes[i].legend(fontsize=8, loc="upper right")

# Remove any empty subplots
for j in range(len(variables_to_plot), len(axes)):
    fig.delaxes(axes[j])

# Set the x-axis label for all subplots
for ax in axes[-n_cols:]:
    ax.set_xlabel('Time (seconds)')

plt.tight_layout()
plt.show()

#"""
#[2268:3813] valeries
#[1262:2234] masha
###
### Plot Cac and ATPc ratio in time
###
#"""



fig2, ax = plt.subplots(figsize = (8, 5))
plt.title('Cac vs ATPc')
"""
tg= res.t[downlim:uplim]
cacg=res.y[5][downlim:uplim]
atpg=res.y[3][downlim:uplim]
"""
tg= res.t
cacg=res.y[5]
atpg=res.y[3]
# using the twinx() for creating another
# axes object for secondary y-Axis
ax2 = ax.twinx()
ax.plot(tg, cacg, color = 'red')
ax2.plot(tg, atpg, color = 'blue')
 
# giving labels to the axises
ax.set_xlabel('Time [s]', color = 'black')
ax.set_ylabel('Cac mM', color = 'red')
 
# secondary y-axis label
ax2.set_ylabel('ATPc mM', color = 'blue')
 
# defining display layout 
plt.tight_layout()
 
# show plot
plt.show()

#"""

#[2268:3813] valeries
###
### Plot ATP:ADP ratio in time
###
"""
fig3, ax = plt.subplots(figsize = (8, 5))
plt.title('ATPm : ADPm')

ratio=res.y[4][downlim:uplim]/res.y[1][downlim:uplim]
tg= res.t[downlim:uplim]

# using the twinx() for creating another
# axes object for secondary y-Axis
ax2 = ax.twinx()
ax.plot(tg, ratio, color = 'red')

# giving labels to the axises
ax.set_xlabel('time', color = 'black')
ax.set_ylabel('ratio', color = 'red')
 
 
# defining display layout 
plt.tight_layout()
 
# show plot
plt.show()
"""

#"""

#[2268:3813] valeries
###
### Plot ATP:ADP ratio in time
###
#"""
fig4, ax = plt.subplots(figsize = (8, 5))
plt.title('ATPc : ADPc')
"""
ratio=res.y[3][downlim:uplim]/res.y[0][downlim:uplim]
tg= res.t[downlim:uplim]
"""
ratio=res.y[3]/res.y[0]
tg= res.t


# using the twinx() for creating another
# axes object for secondary y-Axis

ax.plot(tg, ratio, color = 'red')

# giving labels to the axises
ax.set_xlabel('time', color = 'black')
ax.set_ylabel('ratio', color = 'red')
 
 
# defining display layout 
plt.tight_layout()
 
# show plot
plt.show()

#"""
###
### Plot all concentrations
###
#"""

flux_to_plot = [Jpdh_v, Jcs_v, Jaco_v,  Jidh_v, Jkgdh_v, Jsl_v, Jsdh_v, Jfh_v, Jmdh_v, Jerout_v, Jserca_v, Juni_v, Jncx_v, Jf1_v, Jhl_v, Jant_v, Jo_v, Jhyd_v]



uplimj= len(Jpdh_v)
downlimj= round(len(Jpdh_v)-0.02*len(Jpdh_v))

#Correction of variables from sofia
variables_flux = {
    "Jpdh": 0, "Jcs ": 1, "Jaco ": 2, "Jidh": 3, "Jkgdh": 4, "Jsl": 5,
    "Jsdh": 6, "Jfh": 7,"Jmdh": 8, "Jerout": 9,
    "Jserca" :10,"Juni": 11,"Jncx": 12, "Jf1": 13, "Jhl": 14,
    "Jant": 15, "Jo": 16, "Jhyd": 17
}


# Example: Choose which variables to plot using their dictionary keys
variables_flux_to_plot = list(variables_flux.keys())  # Plotting all variables
steps= np.arange(0,len(Jpdh_v),1)

# Function to extract unit from the variable name
def extract_unit_f(variable_name):
    parts = variable_name.split()
    return parts[-1] if len(parts) > 1 else 'a.u.'  # 'a.u.' for arbitrary unit if no unit is provided

# Determine the number of rows and columns for subplots
n_cols_f = 3
n_rows_f = (len(variables_flux_to_plot) + n_cols_f - 1) // n_cols_f  # Calculate required rows


# Create subplots
fig5, axes = plt.subplots(n_rows_f, n_cols_f, figsize=(17, 1.5 * n_rows), sharex=True)
# Flatten axes for easier iteration

# Flatten axes for easier iteration
axes = axes.flatten()


# Plot each chosen variable
for i, var in enumerate(variables_flux_to_plot):
    var_index = variables_flux[var]
    axes[i].plot(steps[downlimj:uplimj], flux_to_plot[var_index][downlimj:uplimj], label=f'{var}')
    axes[i].legend(fontsize=8, loc="upper right")


# Remove any empty subplots
for j in range(len(variables_flux_to_plot), len(axes)):
    fig4.delaxes(axes[j])


# Set the x-axis label for all subplots
for ax in axes[-n_cols_f:]:
    ax.set_xlabel('Steps')

plt.tight_layout()
plt.show()



