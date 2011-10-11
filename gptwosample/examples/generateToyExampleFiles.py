import scipy as SP, csv, pylab as PL
from scipy import special

def writeBackRepsAddNoise(wc1,wc2,y1,y2,geneName,n_reps):
    for i in range(n_reps):
        c1 = [geneName]
        c2 = [geneName]
        c1.extend(y1+SP.randn(y1.shape[0])*.1)
        c2.extend(y2+SP.randn(y2.shape[0])*.1)
        wc1.writerow(c1)
        wc2.writerow(c2)
    
c1f = open("ToyCondition1.csv",'wb')
c2f = open("ToyCondition2.csv",'wb')
wc1 = csv.writer(c1f,delimiter=",")
wc2 = csv.writer(c2f,delimiter=",")

#HEader with time points
xg = SP.linspace(0,SP.pi,12)
x = SP.arange(0,12,1)
header=["Time [h]"]
header.extend(x)

wc1.writerow(header)
wc2.writerow(header)

#Sixth Gene:
geneName="gene 1"
y1 = -special.erf(SP.sin(2*xg))
y2 = -special.erf(SP.sin(2*xg+1))
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#First Gene:
geneName="gene 2"
y1 = special.erf(SP.sin(.5*xg) * SP.cos(xg))
y2 = y1+.1*SP.exp(xg)*SP.sin(xg-.5)
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#second Gene:
geneName="gene 3"
y1 = SP.sin(xg+.5*SP.pi) * SP.cos(2*xg)
y2 = y1+.1*SP.exp(xg)-.4
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#Third Gene:
geneName="gene 4"
y1 = special.erf(xg)-SP.sin(xg)
y2 = SP.log(y1+.1*SP.exp(xg[::-1]))+1
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#Third Gene:
geneName="gene 5"
y1 = SP.sin(xg)
y2 = SP.sin(xg-1)
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#Fourth Gene:
geneName="gene 6"
y1 = -special.erf(SP.sin(2*xg+.5*SP.pi))
y2 = (.05 * y1)-.78
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

#Fifth Gene:
geneName="gene 7"
y1 = special.erf(xg)-SP.sin(xg)
y2 = y1
writeBackRepsAddNoise(wc1, wc2, y1, y2, geneName, 4)

c1f.close()
c2f.close()
