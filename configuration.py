import numpy as np

#define coord system origin as the centre of the bottom plate
#Find base plate attachment locations
bAngles = np.array([0, 60, 120, 180, 240, 300])
#bAngles = np.array([180, 300, 60])#, 240])
bAngles = np.array([-10, 10, 110, 130, 230, 250])
bAngles = np.radians(bAngles)
bR = 55/1000
bPos = np.array([[bR*np.cos(theta), bR*np.sin(theta), 0] for theta in bAngles])

# Define platform 
#pAngles = 30+np.array([0, 60, 120, 180, 240, 300])
pAngles =  60+np.roll(np.array([-10, 10, 110, 130, 230, 250]), 1)
#pAngles = np.roll(np.arange(0, 360, 360/6), 1) + 30
#pAngles = np.array([180, 0])#, 240])
pAngles = np.radians(pAngles)
#pAngles = bAngles# + np.radians(30)
pR = 52.811/1000
pPos = np.array([[pR*np.cos(theta), pR*np.sin(theta), 0] for theta in pAngles])

legYawAngle = bAngles# + np.radians(90)

# Define length of legs
numLegs = pPos.shape[0]
legLower = np.full(numLegs, 57/1000)
legUpper = np.full(numLegs, 155/1000)