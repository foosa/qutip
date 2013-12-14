#
#
#

from qutip import *
from pylab import *

class Laser:
    """A class to hold laser related parameters"""
    
    def __init__(self, **kwargs):
        
        requiredKeys = {
            "Amplitude" : 1.0,
            "Frequency" : 1.0,
            "Phase" : 0
            }
        
        # Parse user supplied keywords
        for key in kwargs:
            if not hasattr(self, key):
                setattr(self, key, kwargs[key])
                
        # Use default keys if no key was provided
        for key in requiredKeys:
            if not hasattr(self, key):
                setattr(self, key, requiredKeys[key])
                
    def __call__(self, t):
        """Returns value of electric field at time `t`"""
        return self.field(t)
    
    def field(self, t):
        """Returns value of electric field at time `t`"""
        phase = -self.Frequency*t + self.Phase
        return self.Amplitude * cos(phase)
                

def ElectricField(t, lasers):
    """Returns the value of the electric field at time `t` from a collection of
    lasers.
    """
    field = 0
    for laser in lasers:
        field += laser.field(t)
    return field


def SquaredField(t, lasers):
    """Returns the square of the electric field, neglecting the component that 
    rotates at twice the optical frequency.  This leaves only terms that rotate
    at the Raman difference frequency.
    """
    squaredField = 0
    for laserA in lasers:
        for laserB in lasers:
            phase = - (laserA.Frequency - laserB.Frequency)*t + \
                (laserA.Phase - laserB.Phase)
            squaredAmplitude = laserA.Amplitude * laserB.Amplitude
            squaredField += (squaredAmplitude/2.0) * cos(phase)
    
    return squaredField


def AdiabaticEliminate(H, excitedStates):
    """Returns an effective Hamiltonian with the excited state `excitedStates` 
    eliminated from the equations of motion.
    """
    # Find ground states
    states = range(H.shape[0])
    groundStates = []
    for state in states:
        if not state in excitedStates:
            groundStates.append(state)
            
    def SelectBlock(states):
        """Subroutine to select subspaces of matrices"""
        rowSelection = []
        colSelection = []
        for state in states:
            rowSelection.append([state])
            colSelection.append(state)
        return rowSelection, colSelection
    
    # Block-decompose Hamiltonian into components
    rowExcited, colExcited = SelectBlock(excitedStates)
    rowGround, colGround = SelectBlock(groundStates)
     
    HEE = Qobj( H[rowExcited, colExcited] )
    HGE = Qobj( H[rowExcited, colGround] )
    HEG = Qobj( H[rowGround, colExcited] )
    HGG = Qobj( H[rowGround, colGround] )
    
    # Compute effective Hamiltonian
    inverseHEE = Qobj( inv(HEE.full()) )
    HEFF = HGG - HEG * inverseHEE * HEG.dag()
    
    return HEFF


if __name__ == "__main__":
    
    # Frequency splittings
    omegaHyperfine = 0.0#1.0
    omega = 20.0
    
    # Atomic Hamiltonian component
    hamiltonianAtomic = Qobj([
        [omegaHyperfine/2.0, 0, 0],
        [0, -omegaHyperfine/2.0, 0],
        [0, 0, omega]
    ])
    
    # Raman lasers.  The difference frequency is chosen to match the hyperfine
    # frequency splitting that separates states `|0 >` and `|1 >`
    detuning = 19.0
    lasers = [
        Laser(
            Amplitude = 0.1,
            Frequency =  omega - detuning - omegaHyperfine/2.0,
            Phase = 0.0
        ),
        Laser(
            Amplitude = 0.0,
            Frequency = omega - detuning + omegaHyperfine/2.0,
            Phase =0.0
        )
    ]
    
    # Electric dipole operator
    dipole = Qobj([
        [0, 0, 1.],
        [0, 0, 1.],
        [1., 1., 0]
    ])
    
    # *************************************************************************
    # Compute the exact Hamiltonian, without using adiabatic elimination or a
    # rotating-wave approximation.
    
    hamiltonianExact = [
        hamiltonianAtomic,
        [
            - dipole,
            lambda t, args: ElectricField(t, lasers)
        ]
    ]
    
    # *************************************************************************
    # Compute the adiabatic eliminated effective Hamiltonian.  We neglect the
    # anti-Raman term but keep the counter-rotating terms at twice the hyperfine
    # splitting frequency.
    
    # Generator of the interaction frame transformation used to bring the
    # Hamiltonian into the appropriate form for adiabatic elimination.
    frameGenerator = Qobj([
        [omegaHyperfine/2.0, 0, 0],
        [0, -omegaHyperfine/2.0, 0],
        [0, 0,  omega - detuning]
    ])
    
    excitedStates = [2]
    groundStates = [0, 1]
    hamiltonianAdiabatic = AdiabaticEliminate(
        hamiltonianAtomic - frameGenerator - dipole,
        excitedStates
    )
    
    #
    # We have done Adiabatic elimination on the excited states, but left off the
    # time-dependence due to the frame transformation due to the
    # `frameGenerator`.  Here we put this dependence back in term by term ...
    #
    
    frameGeneratorEliminated = AdiabaticEliminate(frameGenerator, excitedStates)
    hamiltonianEliminated = [frameGeneratorEliminated]
    #    AdiabaticEliminate(frameGenerator, excitedStates)
    #]
    print frameGeneratorEliminated
    for final in groundStates:
        for initial in groundStates:
            
            sf = basis(2, final)
            si = basis(2, initial)
            
            hTerm = sf * sf.dag() * hamiltonianAdiabatic * si * si.dag()
            frameShift = frameGenerator[final,final] - frameGenerator[initial,initial]
            #func = lambda t, args: exp(-1j * t * frameShift) * SquaredField(t, lasers)
            #func = lambda t, args: SquaredField(t, lasers)
            func = lambda t, args: (ElectricField(t, lasers))**2
            
            # Append [hTerm, func] pair to the Hamiltonian
            hamiltonianEliminated.append(
                [hTerm, func]
            )
    """
    hamiltonianEliminated = [
        [
            AdiabaticEliminate(
                hamiltonianAtomic - frameGenerator - dipole,
                excitedStates
            ),
            lambda t, args: SquaredField(t, lasers)
        ]
    ]
    """
    
    # *************************************************************************
    # Compute the full dynamics
    print("Computing the full dynamics ...")
    initialState = basis(3,0)
    times = linspace(0, 1500, 100000)
    
    projectors = [
        basis(3,0) * basis(3,0).dag(),
        basis(3,1) * basis(3,1).dag(),
        basis(3,2) * basis(3,2).dag()
    ]
    
    outputExact = mesolve(
        hamiltonianExact,
        initialState,
        times,
        [],
        expt_ops = projectors
    )
    
    # *************************************************************************
    # Compute the Raman approximate dynamics
    print("Computing the Raman approximate dynamics ...")
    initialState = basis(2,0)
    
    projectors = [
        basis(2,0) * basis(2,0).dag(),
        basis(2,1) * basis(2,1).dag()
    ]
    
    outputEliminated = mesolve(
        hamiltonianEliminated,
        initialState,
        times,
        [],
        expt_ops = projectors
    )
    
    
    # *************************************************************************
    # Compute the analytical Raman approximate dynamics
    '''
    print("Computing the analytical Raman approximate dynamics ...")
    initialState = basis(2, 0)
    
    hamiltonianAtomicEliminated = Qobj([
        [omegaHyperfine/2.0, 0],
        [0, -omegaHyperfine/2.0]
    ])
    
    hamiltonianDipoleEliminated = Qobj([
        [1, 1],
        [1, 1]
    ])
    
    hamiltonianAnalytic = [
        hamiltonianAtomicEliminated,
        [
            hamiltonianDipoleEliminated,
            lambda t, args: - SquaredField(t, lasers) / detuning
        ]
    ]
    
    outputAnalytic = mesolve(
        hamiltonianAnalytic,
        initialState,
        times,
        [],
        expt_ops = projectors,
        args = []
    )
    '''
    # *************************************************************************
    # Plotting 
    print("Plotting results ...")
    
    def RamanRabiFrequency():
        """Method to compute the Raman Rabi frequency"""
        freq = 0
        for laserA in lasers:
            for laserB in lasers:
                freq += laserA.Amplitude * laserB.Amplitude / detuning       
        return freq
        
    
    plot(
        times, real( outputExact.expect[0] ), 'r-',
        times, real( outputExact.expect[1] ), 'r-.',
        times, real( outputExact.expect[2] ), 'g',
        times, real( outputEliminated.expect[0] ), 'b-',
        times, real( outputEliminated.expect[1] ), 'b-.',
        #times, real( outputAnalytic.expect[0] ), 'y-',
        #times, real( outputAnalytic.expect[1] ), 'y-.',
        #times, 1/2. * (cos( RamanRabiFrequency() * times ) + 1), 'k'
    )
    xlabel("Time")
    ylabel("Population")
    print("Done ...")
    show()