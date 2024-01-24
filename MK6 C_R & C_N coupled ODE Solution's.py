import numpy as np
import math as m
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%%

'Estimated Parameters Based Upon Research'
# The normal light intensity in a SLA printer is around 5-10 mW/cm^2 or 50Jcm^-2
# KI_0 is normally 0.0015 to 1s^-1
# Parameter definitions
k_blue = 20  # Rate constant for blue light absorption
k_P = 0.02 # Rate constant for P (Solid production) production
k_L = 0.2 # Rate constant for L (Null Substance) Production
I_0_blue = 50 #50Jcm^-2 Intensity of blue light
I_0_UV = 2.5 #50Jcm^-2 Intensity of UV light
mu_bar = 3.74  # Absorption coefficient
k_UV = 0.002  # Rate constant for UV light absorption


#%%

# Define the initial conditions
CR_0 = 0  # Initial concentration of CR
CM_0 = 1  # Initial concentration of CM
CN_0 = 0  # Initial concentration of CN
CP_0 = 0  # Initial concentration of CP
CPA_0 = 1 # Initial concentration of CPA
CPB_0 = 1 # Initial concentration of CPB

#%%

CM_vec = []

'Coupled ODE solving function'
# Function to compute the ODEs
def dCR_dz_dt(t, y, z):
    # Extract the current concentrations
    # see: https://uk.mathworks.com/help/matlab/math/choose-an-ode-solver.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    # e.g., https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    
    ''' y contains both CP and CR, split up accordingly''' 
    #CR and CP are defined first for used in later equations for the numerical intergration. 
    CR = y[0:len(z)] # 0 up to element before `len(z)`. This creates a vector of all the CR values. len(z) being the total length of the vector for 
    #all the z's, which are used in generating the values for CR. 
    CP = y[len(z) : ] # all elements from `len(z)` up to `end` of vector. The end of the values for CR, all the way to the total end of the CP values. 
   
    CM = CM_0 - CP # Compute CM based on the equation CM = CM_0 - CP
    CM_vec.append(CM)
    
    I_blue = I_0_blue * np.exp(-mu_bar * z) # <- impose `z` in element wise manner. All values are solved simultaeously in element wise manner.
    I_UV = I_0_UV * np.exp(-mu_bar * z)
    
    CPA = CPA_0 * np.exp(-k_blue * I_blue * t)
    CPB = CPB_0 * np.exp(-k_UV * I_UV * t)
    
    CN = CPA + CP + CR - CPB - CPA_0 + CPB_0
    
    # Compute the rates of change for each component of the system
    # CR and CP are coupled, so must be solved together, you could separate these into 
    # a 2 helper functions, e.g. CR(...) and CP(...) , but then you must call them within here
    dCR = I_blue * k_blue * CPA  - k_P * CR * CM - k_L * CR * CN
    dCP = k_P * CR * (CM_0 - CP)
    
    'Concatinate both vectors of solutions'
    #This is done to speed up computation.
    dTotal = np.concatenate((dCR, dCP),
                        axis = None)
    '''
    must rejoin because we have defined `y` vector as:
    y_1 = CR(z=z1, t)
    y_2 = CR(z=z2, t) 
    ...
    y_{N_z} = CR(z=zN, t)
    y_{N_z + 1} = CP(z=z1, t)
    y_{N_z + 1} = CP(z=z2, t)
    ...
    y_{N_z + N_z} = CP(z=zN, t)
    
    Therefore by our definiton of the 1st order variable (i.e. we performed a transform, and applied method of lines...)
    we must repackage dy/dt accordingly, which this function returns the values for
    i.e.
    dTotal_1 = dy_1/dt (= f_1( vec{y}, t) )
    ...
    etc.
    
    '''    
    'Return the total Solution'
    # rejoin as a concatanted vector (i.e. back to (12,1) or whatever it is)
    return dTotal#[dCR, dCP]

#%%

# Define the time interval
t_span = (0.0, 43200)
t_eval = np.linspace(t_span[0], t_span[1],50000)


#%%

# Define the z values
z_values = np.arange(0, 15, 0.1)  # Array of z values ranging from 0 to 2 with a step of 0.1

#%%

'Vector size determination'
numel_z = len(z_values) #Interger numbering the amount of elements in z_values
'Line 101 Not in use'
numel_y = 2 * numel_z #Interger numbering two times the amount of elements in z_values, so it can be used to create a vector that will
#fit both CR and CP values within.
# combine CR and CP variables under the 1st order variable `u`
#y0 = [CR_0, CP_0] -> [ones(numel_z) * CR_0, ones(numel_z) * CP_0]

#%%

'Vectors of initial conditions creation'
# Generate a vector of initial values for CR and CP, then join them for the initial guess vector under `y0`
# populating a vector with our initial conditions 
CR_0_vector = CR_0 * np.ones(numel_z) #Creates vector the size of z_values, filled with the intial condition of CR
# CR_0 {scalar} \times a vector of `1` of length `numel_z`
CP_0_vector = CP_0 * np.ones(numel_z) #Creates vector the size of z_values, filled with the intial condition of CP
CM_0_vector = CM_0 * np.ones(numel_z)

y0 = np.concatenate( (CR_0_vector, CP_0_vector),#Inital condition vectors are concatinated into one vector for computation within the solver function.
                    axis = None)

#%%

#dCR_dz_dt(0, y0, z_values) # test line - see if `dCR_dz_dt` works; use with a break point

'Solve the coupled ODEs with solve_ivp'
solution_total = solve_ivp(fun=lambda t, y: dCR_dz_dt(t, y, z_values), 
                           t_span=t_span, 
                           y0=y0, 
                           t_eval=t_eval, dense_output=True)
                           
'Unpack solutions for manipulation'
# as before, unpack the ([Vector length of both CR and CP],:) size into corresponding ([Vector length of either CR or CP],:)
#size for CP and CR as before.
# But this time it's for the solutions, so they can be graphed                         
sol_CR = solution_total.y[0:numel_z, :] # get time series for all CR at given z locations
sol_CP = solution_total.y[numel_z:, :] # get time series for all CP at given z locations
# CR: [0 to the element just before `numel_z`, all the time values]
# CP: [`numel_z` to the last entry in the row, all the time values] (time values in the columns...)

#%%

'Create figures for Graphs'
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

#%%

'Plot graph for CP, CR vs Time, with incrementing values of z'

color_CR = 'blue'
color_CP = 'red'
color_CN = 'green'

time_red = [0, 37.5, 75, 150, 300, 600, 1800, 3600, 5400, 7200, 9000, 10800, 21600, 43200]
#time_red = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#i is the index of the time_red, whilst time_x is the value at that index.
for i, time_x in enumerate(time_red):
    #for i, z_0 in enumerate(solution_total.sol(time_x)):#z_values):
    label_CR = 'C_R' if i == 0 else None  # Set the label for C_R only in the first iteration to avoid duplicate legends
    label_CP = 'C_P' if i == 0 else None  # Set the label for C_M only in the first iteration to avoid duplicate legends

    'The solutions unpacked of explicit times specified'
    # use `.sol` interpolation feature to evalute solution at time `time_x`
    Solution_time_x = solution_total.sol(time_x) #Solutions are unpacked for explicit times in time_red
    #unpack as before...
    #Remember CR is at the beginning of the vector y, so must be unpackaged and separated. 
    sol_CR_current = Solution_time_x[0:numel_z] # Specific solutions for CR at time_x, for all CR at given z locations
    sol_CP_current = Solution_time_x[numel_z:] # Specific solutions for CP at time_x, for all CP at given z locations
    
    'This is the equation for conversion fraction'
    ϕ = sol_CP_current/CM_0_vector
    
    ax1.plot(z_values, ϕ,'-', label=label_CP, color=color_CR)  # Plot C_R vs. Time
    
'Add an arrow to the plot'
arrow_start = (0, 0)
arrow_end = (3, 0.7)
arrow_props = dict(arrowstyle='->', color='black')
# Add the arrow to the plot
ax1.annotate('', arrow_end, arrow_start, arrowprops=arrow_props)

# Add the text at the endpoint
text_position = arrow_end
text_offset = (3, 0.7)  # Adjust the offset as needed
ax1.annotate('Time (s), t', xy=text_position, xytext=text_offset)


ax1.set_xlabel('Polymerization Displacement (mm), z  ')
ax1.set_ylabel('Conversion Fraction, ϕ')
ax1.set_title('C_P vs. Polymerization Displacement z')
ax1.legend()

#%%
'Plot graph for CP, CR vs z, with incrementing values of Time'
#i is the index of the z_values, whilst z_0 is the value at that index.
for i, z_0 in enumerate(z_values):
    label_CR = 'C_R' if i == 0 else None  # Set the label for C_R only in the first iteration to avoid duplicate legends
    label_CP = 'C_P' if i == 0 else None  # Set the label for C_M only in the first iteration to avoid duplicate legends

    # plot sol_CR / sol_CP time-series - this should be plotting row `i` and all the column values
    # sol_CR is pulling the solutions from CR for each indexed row, and all columns of that row of the matrix of CR's for the specified z value. 
    #The new z indexes being the new rows of the solutions for the individuals times of the columns. 
    ax2.plot(solution_total.t, sol_CR[i,:], label=label_CR, color=color_CR)  # Plot C_R vs. Time
    ax2.plot(solution_total.t, sol_CP[i,:], label=label_CP, color=color_CP)  # Plot C_M vs. Time

    ax2.set_xlabel('Time (s), t')
    ax2.set_ylabel('Concentration (moles/m^3), C')
    ax2.set_xscale('log')
    ax2.set_xlim(0.1, solution_total.t.max())
    ax2.set_title('C_R and C_P vs. Time t')
    ax2.legend()