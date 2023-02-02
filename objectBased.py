
import numpy as np

import rotary

class RotaryStewartPlatform():
    def __init__(self):
        #define coord system origin as the centre of the bottom plate
        #Find base plate attachment locations
        self.bAngles = np.array([0, 60, 120, 180, 240, 300])
        self.bAngles = np.array([-10, 10, 110, 130, 230, 250])
        self.bAngles = np.radians(self.bAngles)
        self.bR = 100/1000
        self.bPos = np.array([[self.bR*np.cos(theta), self.bR*np.sin(theta), 0] for theta in self.bAngles])

        # Define platform 
        self.pAngles =  60+np.roll(np.array([-10, 10, 110, 130, 230, 250]), 1)
        self.pAngles = np.radians(self.pAngles)
        self.pR = 160/1000
        self.pPos = np.array([[self.pR*np.cos(theta), self.pR*np.sin(theta), 0] for theta in self.pAngles])

        self.legYawAngle = self.bAngles

        # Define length of legs
        self.numLegs = self.pPos.shape[0]
        self.legLower = np.full(self.numLegs, 60/1000)
        self.legUpper = np.full(self.numLegs, 150/1000)



        # The pose variables are:
        """translation"""
        self.trans = [0, 0, 0]
        """angles"""
        self.angles_rad = [0, 0, 0]
        """midJoint"""
        self.midJoint = np.full((self.numLegs, 3), np.nan)
        """leverAngles"""
        self.leverAngles = self.numLegs * [np.nan]

        trans_init = [0, 0, 1.02574227e-1]
        angles_init_rad = [0, 0, 0]
        self.inverse(trans_init, angles_init_rad)



    def inverse(self, translation, angles):
        #translation = [0.0, 0, 1.02574227e-01]  # 0.13 is about neutral
        #angles = list(np.radians([0, 0, 0]))
        upperNew = rotary.ik(np.array([0, 0, 0]), self.pPos, self.legLower, self.legUpper, translation+angles)

        self.tran = upperNew[:3]
        self.angles_rad = upperNew[3:]

        self.midJoint, self.leverAngles = rotary.legPos(self.bPos, upperNew, self.legLower, self.legUpper, self.legYawAngle)
        for jointIndex, pos in enumerate(self.midJoint):
            # We want to always choose the solution that has the joint outside the stewart platform
            #    TODO: Figure out why this check does that (what coordiante system is midJoint in?)
            if pos[2] < 0:
                self.leverAngles[jointIndex] *= -1

        # TODO: Put some of these back in as asserts
        # TODO: Log this
        #print("Lower Leg Length", np.sqrt(np.sum(np.square(self.midJoint-self.bPos), 1)))
        #print("Upper Leg Length", np.sqrt(np.sum(np.square(self.upperNew-self.midJoint), 1)))
        #print("Lower Leg Angles", np.degrees(self.leverAngles))

    def forward(self, leg_angles_rad):
        # Assume the last calculated position as a decent starting point
        #   for the iterative solver
        trans_init_guess = self.trans
        angles_init_guess_rad = self.angles_rad
        # Then solve
        translation, angles, solnInfo = rotary.fkWorking(self.bPos,
                                                        self.pPos,
                                                        self.legLower,
                                                        self.legUpper,
                                                        self.legYawAngle,
                                                        leg_angles_rad,
                                                        trans_init_guess,
                                                        angles_init_guess_rad)



    def leg_forces(self, force, moment):
        # Calculate force
        #moment = np.array([0, 0, 0])
        #force = np.array([0, 0, 10*9.81])
        f2, A, b = rotary.forces(self.bPos, list(self.trans) + list(self.angles_rad), moment, force)
        #print(f2)

        return f2

    def leg_torques(self, force, moment):
        f2 = self.leg_forces(force, moment)

        lt = rotary.legTorques(self.bPos, self.midJoint, list(self.trans) + list(self.angles_rad), self.legLower, f2)
        print("Hence")
        print(lt)
        print(lt*100/9.81)

        return lt

    def platform_forces(self, leg_torques):
        ###############################################################################
        # Resolving forces to platform forces
        ###############################################################################
        # Assume we know:
        # * The pose of the stewart platform
        # * The component of the force perpendicular to the lower leg (ie. the torque component)
        legMeasForce_N = (leg_torques.reshape(1, -1)/self.legLower).T
        print(legMeasForce_N)

        # Hence can fill this back out to the force in the lower leg.
        lowerLegVecs = self.midJoint - self.bPos
        lowerLegDirVecs = rotary.normVec(lowerLegVecs)
        for legNum in range(self.numLegs):
            print(f"\nLeg: {legNum}")
            
            legForceComponent = legMeasForce_N[legNum]
            
            # Now, we know this is force in the direction perpendicular - ie.
            #    the (local) y coordinate?
            
            
            
            # TODO: Rigour in coordinate systems
            legForceMagnitude = legForceComponent/lowerLegDirVecs[legNum, 1]  # TODO: is 1 the correct bit?
            legForce = legForceMagnitude * lowerLegDirVecs[legNum, :]
            print(legForce)

