


deplog = [0.96, 0.91, 0.93]
loganomaly = [0.94, 0.97, 0.95]
robustlog = [0.92, 0.98, 0.94]
ourMethod = [0.95, 0.99, 0.96]

f1deplog = 2*(deplog[0]*deplog[1])/(deplog[0]+ deplog[1])
print(f1deplog)

f1loganomaly = 2*(loganomaly[0]*loganomaly[1])/(loganomaly[0] + loganomaly[1])
print(f1loganomaly)

f1robustlog = 2*(robustlog[0]*robustlog[1])/(robustlog[0] + robustlog[1])
print(f1robustlog)

f1ourMethod = 2*(ourMethod[0]*ourMethod[1])/(ourMethod[0] + ourMethod[1])
print(f1ourMethod)