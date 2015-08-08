'''
Created on Aug 8, 2015
@author: Jonas Wallin
'''
try:
	from setuptools import setup, Extension
except ImportError:
	try:
		from setuptools.core import setup, Extension
	except ImportError:
		from distutils.core import setup, Extension
		
metadata = dict(name='MCMCPYJW',
	packages         = ['MCMCPYJW'],
	package_dir      = {'MCMCPYJW': 'MCMCPYJW'},
	version          = '0.1',
	description      = 'various MCMC and AMCMC scripts',
	author           = 'Jonas Wallin',
	maintainer_email = 'jonas.wallin81@gmail.com',
	url              = 'https://github.com/JonasWallin/MCMCPYJW',
	author_email     = 'jonas.wallin81@gmail.com',
	install_requires = ['numpy'])
setup(**metadata)