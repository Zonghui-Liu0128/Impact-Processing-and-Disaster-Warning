import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential',
                 atmos_filename='./armageddon/resources/AltitudeDensityTable.csv',
                 Chely_filename='./armageddon/resources/ChelyabinskEnergyAltitude.csv',
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'

        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option

        Chely_filename : string, optional
            Name of the filename to use with the curve-fitting of dedz

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.atmos_filename = atmos_filename
        self.Chely_filename = Chely_filename
        
        try:
            # set function to define atmoshperic density
            if atmos_func == 'exponential':
                self.rhoa = lambda x: self.rho0 * np.exp(-x/self.H)
            elif atmos_func == 'tabular':

                self.df = pd.read_csv(self.atmos_filename,
                                  skiprows=6, header=None, sep=' ').to_numpy()
                def rhoa(z):
                    z = np.array(z)
                    i = np.where((z/10.).astype(np.int)<8600, (z/10.).astype(np.int), 8600)
                    z_i = self.df[i, 0]
                    rho_i = self.df[i, 1]
                    H_i = self.df[i, 2]
                    return rho_i*np.exp((z_i-z)/H_i)
                self.rhoa = rhoa
                
            elif atmos_func == 'constant':
                self.rhoa = lambda x: rho0
            else:
                raise NotImplementedError(
                    "atmos_func must be 'exponential', 'tabular' or 'constant'")
        except NotImplementedError:
            print("atmos_func {} not implemented yet.".format(atmos_func))
            print("Falling back to constant density atmosphere for now")
            self.rhoa = lambda x: rho0


    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        Returns
        -------
        Result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'
        """

        # Enter your code here to solve the differential equations
        def f(t, y):
            """
            Calculate the variation of necessary elements between each step

            Parameters
            ----------
            y : array
                The list contains values of necessary elements
            
            Returns
            -------
            Result : array
                A array contains the variation of those necessary elements.
                Includes the following elements:
                'velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius'
            """
            rhs = np.zeros_like(y)
            rhoa = self.rhoa(y[3])
            A = np.pi * (y[5]**2)
            # v
            rhs[0] = - (self.Cd * rhoa * A * (y[0]**2))/(2*y[1]) + self.g * np.sin(y[2])
            # m
            rhs[1] = - (self.Ch * rhoa * A * (y[0]**3))/(2*self.Q)
            # angle
            rhs[2] = (self.g * np.cos(y[2])) / y[0] - (self.Cl * rhoa * A * y[0])/(2 * y[1]) - (y[0] * np.cos(y[2]))/(self.Rp+y[3])
            # z
            rhs[3] = -y[0] * np.sin(y[2])
            # x
            rhs[4] = (y[0] * np.cos(y[2]))/(1 + (y[3] / self.Rp))
            rhs[5] = 0
            if rhoa*(y[0]**2) >= strength:
                rhs[5] = ((7./2.) * self.alpha * rhoa/density)**0.5 * (y[0])
           
            return rhs
            
        def RK4(f, y0, t0, tend, dt):
            """
            Solve the given set of ODEs
            Implement by RK4 method

            Parameters
            ----------
            f: function
                Derivative of each element

            y_0: array
                Start initial value

            t_0: float
                Start time

            tend: float
                End time

            dt : float
                The time step

            strength : float
                The strength of the asteroid (i.e., the ram pressure above which
                fragmentation and spreading occurs) in N/m^2 (Pa)

            density : float
                The density of the asteroid in kg/m^3

            Returns
            -------
            Result : np.array
                y_all = [velocity, mass, angle, altitude, distance, radius]
                t_all = [time]
            """
            y = y0
            t = t0
            y_all = [y0]
            t_all = [t0]                 
            while t < tend:
                k1 = dt*f(t, y)
                k2 = dt*f(t + 0.5*dt, y + 0.5*k1)
                k3 = dt*f(t + 0.5*dt, y + 0.5*k2)
                k4 = dt*f(t + dt, y + k3)
                y = y + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
                
                if (y[0] <= 20) or (y[3] <= 0):
                    break
                y_all.append(y)
                t = t + dt
                t_all.append(t)

            return y_all, t_all
        
        #initial conditions
        r0 = radius 
        rhom = density 
        strength = strength 
        v0 = velocity 
        m0 = rhom * (4/3) * np.pi * (r0**3) 
        angle0 = np.deg2rad(angle) 
        z0 = init_altitude 
        x0 = 0. 

        dt = 0.05
        t0 = 0.0
        tend = 1000
        y0 = np.array([v0,m0,angle0,z0,x0,r0])

        y_rk4, time = RK4(f,y0,t0,tend,dt)
        
        y_rk4 = pd.DataFrame(y_rk4,columns=['velocity','mass','angle','altitude','distance','radius'])
        time = pd.DataFrame(time,columns=['time'])
        
        result0 = pd.concat([y_rk4,time],axis=1) 

        self.result = pd.DataFrame({'velocity': result0['velocity'],
                        'mass': result0['mass'],
                        'angle': np.rad2deg(result0['angle']),
                        'altitude': result0['altitude'],
                        'distance': result0['distance'],
                        'radius': result0['radius'],
                        'time': result0['time']})
        return self.result


    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.

        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time

        Returns : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude

        """

        # Replace these lines with your code to add the dedz column to
        # the result DataFrame
        result = self.result.copy()
        # result.insert(len(result.columns),'dedz', np.array(np.nan))

        result['ek'] = 0.5 * result['mass'] * result['velocity'] ** 2
        result['ek_kt'] = result['ek'] / 4.184e12
        result['z_km'] = result['altitude'] * 0.001

        shifted = result.shift(1)
        result['dedz'] = (shifted['ek_kt']-result['ek_kt'])/(shifted['z_km']-result['z_km'])

        dedz = result['dedz'].tolist()
        z_km = result['z_km'].tolist()

        result = result.drop(columns=['ek', 'ek_kt','z_km'])
        self.result = result.fillna(0)
        
     
        # #(Chelyabinsk event)import energy decomposition curve
        # chely_curve = pd.read_csv(self.Chely_filename)
        # z_chely = chely_curve['Height (km)'].tolist()
        # dedz_chely = chely_curve['Energy Per Unit Length (kt Km^-1)'].tolist()
        
        # #plot
        # plt.plot(dedz, z_km, 'k', label='de/dz')
        # plt.plot(dedz_chely,z_chely,'b',label='Energy decomposition curve (Chelyabinsk)')
        # plt.xlabel("Energy per unit height [Kt/km]")
        # plt.ylabel("Altitude [km]")
        # plt.xlim(0)
        # plt.ylim(0,90)
        # plt.legend(loc='best')
        # plt.show()
        return self.result

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats

        Parameters
        ----------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time

        Returns
        -------
        outcome : Dict
            dictionary with details of the impact event, which should contain
            the key ``outcome`` (which should contain one of the following strings:
            ``Airburst`` or ``Cratering``), as well as the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_distance``, ``burst_energy``
        """
        # record the initial value of velocity and mass
        result = self.result.copy()
        v_i = result.velocity[0]
        m_i = result.mass[0]
        # find the burst data 
        burst = result.sort_values('dedz', ascending = False)
        burst1 = burst.head(1) 
        # gain the burst altitude
        burst_list = burst1.altitude.tolist()[0]

        # judge whether the state is airburst or cratering by judging whether the 
        # burst altitude is higher than 0 or not
        if burst_list > 0:
            state = "Airburst"
            burst_peak_dedz = burst1.dedz.tolist()[0]
            burst_altitude = burst1.altitude.tolist()[0]
            burst_distance = burst1.distance.tolist()[0]
            v_burst = burst1.velocity.tolist()[0]
            m_burst = burst1.mass.tolist()[0]
            # burst_energy is gained by calculating the difference between 
            # the initial kinetic energe and the kinetic energe in the burst moment
            burst_energy = np.abs(1/2 * (m_i * v_i**2 - m_burst*v_burst**2) / 4.184e12 )
        else:
            state = "Cratering"
            # take the targeted slice which the planet is close to the ground
            left = result[-0.1<result['altitude']]
            slice = left[left['altitude']<0.1]
            # take the data which the planet is the most near to 0 in the slice
            min = 0.1
            for row in slice.iterrows():
                if abs(row[1]['altitude']) < min:
                    min = abs(row[1]['altitude'])
                    index_number = row[0]
            target_row = slice[slice.index==index_number]
            burst_peak_dedz = target_row.dedz.tolist()[0]
            burst_altitude = 0
            burst_distance = target_row.distance.tolist()[0]
            # choose the larger of the total kinetic energy lost by the asteroid or 
            # the residual kinetic energy of the asteroid when it hits the ground as the burst energy
            v_ground = target_row.velocity.tolist()[0]
            m_ground = target_row.mass.tolist()[0]
            energy_1 = np.abs((1/2 * (m_i * v_i**2 - m_ground * v_ground**2)) / 4.184e12)
            energy_2 = np.abs((1/2 * m_ground * v_ground**2) / 4.184e12)
            burst_energy = np.maximum(energy_1,energy_2)

        outcome = {'outcome': state,
                   'burst_peak_dedz': burst_peak_dedz,
                   'burst_altitude': burst_altitude,
                   'burst_distance': burst_distance,
                   'burst_energy': burst_energy}
        # print(outcome)

        return outcome

