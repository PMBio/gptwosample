'''
Created on Apr 3, 2013

@author: Max
'''
import unittest
from gptwosample.data.dataIO import get_data_from_csv
import numpy
import os

teststring = """input,1,2,3,\n
g1,.3,.4,,.6\n
g1,.3,.4,.5,.6\n
g2,4,3,ignorethis,1\n
g2,4,3,2,1\n
othername,1,2,3,4\n
othername,1,2,3,4\n"""

class Test(unittest.TestCase):

    def setUp(self):
        with open('tmp.csv','w') as f:
            f.write(teststring)
        pass

    def tearDown(self):
        os.remove('tmp.csv')
        pass

    def testRead(self):
        d = get_data_from_csv('tmp.csv')
        for name in ['g1','g2','othername']:
            assert name in d
            assert d[name].shape == (2,4)
        assert numpy.allclose(d["othername"][0],[1,2,3,4])
        assert numpy.all(d['input'][numpy.isfinite(d["input"])] == [1.,2.,3.])
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    #d = get_data_from_csv(teststring)
    unittest.main()