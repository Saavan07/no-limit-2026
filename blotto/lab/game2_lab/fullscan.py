"""Exact full-action sample scoring by a five-dimensional prefix transform."""
import ctypes
from functools import lru_cache
from math import comb
from pathlib import Path
import subprocess
import numpy as np


@lru_cache(maxsize=1)
def library():
    folder=Path(__file__).resolve().parent;source=folder/'fullscan.cpp';target=folder/'fullscan.dylib'
    included=folder.parent/'data/parry-replay/lookup.cpp'
    if not target.exists()or target.stat().st_mtime<max(source.stat().st_mtime,included.stat().st_mtime):
        subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(target)],check=True)
    lib=ctypes.CDLL(str(target));ptr=np.ctypeslib.ndpointer
    u8=ptr(dtype=np.uint8,flags='C_CONTIGUOUS');i32=ptr(dtype=np.int32,flags='C_CONTIGUOUS');i64=ptr(dtype=np.int64,flags='C_CONTIGUOUS');f64=ptr(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.simplex_sample_sums.argtypes=[ctypes.c_int,u8,i32,i64,ctypes.c_int,i64,f64];lib.simplex_sample_sums.restype=ctypes.c_int
    lib.direct_sample_sums.argtypes=[u8,ctypes.c_int,u8,i32,i64,ctypes.c_int,i64,ctypes.c_int]
    lib.direct_sample_matrix4.argtypes=[u8,ctypes.c_int,u8,i32,ctypes.c_int,u8,ctypes.c_int]
    return lib


def checked_groups(groups,counts):
    maxima,codes,_=groups;counts=np.asarray(counts)
    if maxima.ndim!=2 or maxima.shape[1]!=5 or len(maxima)!=len(codes)or counts.shape!=(len(maxima),):raise ValueError('Invalid group shapes')
    if np.any(maxima<0)or np.any(maxima>100)or np.any(maxima!=np.floor(maxima))or np.any(codes<0)or np.any(codes>=16807)or np.any(codes!=np.floor(codes)):raise ValueError('Invalid group statistics')
    if np.any(counts<=0)or np.any(counts!=np.floor(counts))or counts.sum()>2**31:raise ValueError('Positive bounded integer group counts required')
    return np.ascontiguousarray(maxima,dtype=np.uint8),np.ascontiguousarray(codes,dtype=np.int32),np.ascontiguousarray(counts,dtype=np.int64)


def all_sample_sums(groups,counts,budget=100):
    if type(budget)is not int or not 0<=budget<=100:raise ValueError('Budget must be 0 through 100')
    maxima,codes,mass=checked_groups(groups,counts)
    if np.any(maxima>budget):raise ValueError('Opponent maxima exceed the budget')
    result=np.empty(comb(budget+4,4),dtype=np.int64);timing=np.empty(3,dtype=float)
    status=library().simplex_sample_sums(budget,maxima,codes,mass,len(mass),result,timing)
    if status:raise RuntimeError('Simplex scorer failed with status '+str(status))
    return result,dict(zip(['scatter_seconds','prefix_seconds','extract_seconds'],timing.tolist()))


def direct_sums(actions,groups,counts,workers=4):
    x=np.asarray(actions)
    if x.ndim!=2 or x.shape[1]!=5 or np.any(x<0)or np.any(x>100)or np.any(x!=np.floor(x)):raise ValueError('Invalid actions')
    maxima,codes,mass=checked_groups(groups,counts);x=np.ascontiguousarray(x,dtype=np.uint8)
    out=np.empty(len(x),dtype=np.int64)
    library().direct_sample_sums(x,len(x),maxima,codes,mass,len(mass),out,workers)
    return out


def direct_matrix4(actions,groups,workers=4):
    x=np.asarray(actions)
    if x.ndim!=2 or x.shape[1]!=5 or np.any(x<0)or np.any(x>100)or np.any(x!=np.floor(x)):raise ValueError('Invalid actions')
    maxima,codes,_=checked_groups(groups,np.ones(len(groups[0]),dtype=np.int64));x=np.ascontiguousarray(x,dtype=np.uint8)
    out=np.empty((len(x),len(maxima)),dtype=np.uint8)
    library().direct_sample_matrix4(x,len(x),maxima,codes,len(maxima),out,workers)
    return out


@lru_cache(maxsize=2)
def allocations(budget=100):
    if type(budget)is not int or not 0<=budget<=100:raise ValueError('Invalid budget')
    xs=np.empty((comb(budget+4,4),5),dtype=np.uint8);position=0
    for a in range(budget+1):
        for b in range(budget-a+1):
            for c in range(budget-a-b+1):
                rem=budget-a-b-c;size=rem+1
                xs[position:position+size,:3]=[a,b,c];xs[position:position+size,3]=np.arange(size);xs[position:position+size,4]=np.arange(rem,-1,-1)
                position+=size
    xs.flags.writeable=False
    return xs


def allocation_indices(actions,budget=100):
    x=np.asarray(actions)
    if x.ndim!=2 or x.shape[1]!=5 or np.any(x<0)or np.any(x!=np.floor(x))or np.any(x.sum(axis=1)!=budget):raise ValueError('Invalid allocations')
    x=x.astype(np.int64);r=np.full(len(x),budget,dtype=np.int64);index=np.zeros(len(x),dtype=np.int64)
    for j,k in enumerate([4,3,2]):
        table=np.array([comb(n,k)if n>=k else 0 for n in range(budget+5)],np.int64)
        index+=table[r+k]-table[r-x[:,j]+k];r-=x[:,j]
    return index+x[:,3]
