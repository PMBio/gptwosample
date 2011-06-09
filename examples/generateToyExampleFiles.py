import scipy as SP, csv, pylab as PL
from scipy import special

def writeBackRepsAddNoise(wc1,wc2,y1,y2,geneName,n_reps):
    for i in range(n_reps):
        c1 = [geneName]
        c2 = [geneName]
        c1.extend(y1+SP.randn(y1.shape[0])*.2)
        c2.extend(y2+SP.randn(y2.shape[0])*.2)
        wc1.writerow(c1)
        wc2.writerow(c2)
    

wc1 = csv.writer(open("ToyCondition1.csv",'wb'),delimiter=",")
wc2 = csv.writer(open("ToyCondition2.csv",'wb'),delimiter=",")

#HEader with time points
xg = SP.linspace(0,SP.pi,12)
x = SP.arange(0,12,1)
header=["Time [h]"]
header.extend(x)

wc1.writerow(header)
wc2.writerow(header)

#First Gene:
geneName="gene 1"
y1 = special.erf(SP.sin(.5*xg) * SP.cos(xg))
y2 = y1+.1*SP.exp(xg)*SP.sin(xg-.5)
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#second Gene:
geneName="gene 2"
y1 = SP.sin(xg+.5*SP.pi) * SP.cos(2*xg)
y2 = y1+.1*SP.exp(xg)
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#Third Gene:
geneName="gene 3"
y1 = special.erf(xg)-SP.sin(xg)
y2 = SP.log(y1+.1*SP.exp(xg[::-1]))+1
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#Fourth Gene:
geneName="gene 4"
y1 = -special.erf(SP.sin(2*xg+.5*SP.pi))
y2 = (.05 * y1)-.7
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#Fifth Gene:
geneName="gene "
y1 = special.erf(xg)-SP.sin(xg)
y2 = y1
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)
