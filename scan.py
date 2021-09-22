from __future__ import absolute_import
from __future__ import print_function
import pyopencl as cl
import numpy as np
import numpy.linalg as la
import cProfile as cp
import random

class Scan:
    def __init__(self, ctx, queue) :
        self.ctx = ctx
        self.queue = queue
        self.d_grps = list()
        self.g_siz = list()
        self.l_siz = list()
        self.limit = list()
        with open('kernel/scan.cl', 'r') as fp : src = fp.read()
        self.prg = cl.Program(self.ctx, src).build()

    def initMem(self, size) :
        mf = cl.mem_flags
        sm = 64
        while size>(4*sm) :
            gsiz = (size+3)//4
            self.limit.append(np.int32(gsiz))
            self.g_siz.append(((gsiz+sm-1)//sm*sm,))
            self.l_siz.append((sm,))
            size = (gsiz+sm-1)//(sm)
            self.d_grps.append(cl.Buffer(self.ctx, mf.READ_WRITE, (size+1)*4))
        if size :
            self.d_grps.append(None)
            self.g_siz.append((size,))
            self.l_siz.append((size,))
            self.limit.append(size)

        print('self.g_siz : ', self.g_siz)
        print('self.l_siz : ', self.l_siz)
        print('self.limits : ', self.limit)
        print('self.d_grps length : ', len(self.d_grps))

    def run(self, d_dst, d_src) :
        d_srcs = [d_src, ]
        d_dsts = [d_dst, ]
        evt=list()

        for d_grp in self.d_grps :
            d_srcs.append(d_grp)
            d_dsts.append(d_grp)
        print("run:: d_srcs.length : ", len(d_srcs))
        print("run:: d_dsts.length : ", len(d_dsts))

        print("d_srcs : ", d_srcs)
        print("d_dsts : ", d_dsts)
        print("self.d_grps : ", self.d_grps)

        for env in zip(self.g_siz, self.l_siz, d_dsts, d_srcs, self.d_grps, self.limit) :
            if env[4] is not None :
                print(f"scan4 execute :  ")
                print(f"    dispatch_size = {env[0]}")
                print(f"    local_size = {env[1]}")
                print(f"    limits = {env[5]}")
                print(f"    d_src : {env[3]}, d_dst : {env[2]} d_grp : {env[4]}")
                evt.append(self.prg.scan4(self.queue, env[0], env[1], env[2], env[3], env[4], env[5]))
            else :
                print(f"scan_ed execute :  ")
                print(f"    dispatch_size = {env[0]}")
                print(f"    local_size = {env[1]}")
                print(f"    d_src : {env[3]}, d_dst : {env[2]}")
                evt.append(self.prg.scan_ed(self.queue, env[0], env[1], env[2], env[3], cl.LocalMemory(env[0][0]*4*2)))

        for env in reversed(list(zip(self.g_siz, self.l_siz, d_dsts, d_srcs, self.d_grps))) :
            if env[4] is not None :
                print("uniformUpdate execute : ")
                print(f"    dispatch_size = {env[0]}")
                print(f"    local_size = {env[1]}")
                print("     d_dst : ", env[2], " d_grp : ", env[4])
                evt.append(self.prg.uniformUpdate(self.queue, env[0], env[1], env[2], env[4]))
        return evt

def main() :
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(dev_type=cl.device_type.GPU, properties = [ (cl.context_properties.PLATFORM, platforms[0] ) ])
    queue = cl.CommandQueue(ctx, properties = cl.command_queue_properties.PROFILING_ENABLE)
    scan = Scan(ctx, queue)

    arr_size = 63*31*31 *3
    #arr_size = 16

    h_src = np.random.randint(low=100, size=arr_size, dtype=np.int32)
    h_src = np.ones(arr_size, dtype=np.int32)
    h_dst = np.zeros(arr_size, dtype=np.int32)

    mf = cl.mem_flags
    d_src = cl.Buffer(ctx, mf.READ_WRITE, arr_size*4)
    d_dst = cl.Buffer(ctx, mf.READ_WRITE, arr_size*4)
    
    cl.enqueue_copy(queue=queue, dest=d_src, src=h_src)

    scan.initMem(arr_size)
    scan.run(d_dst, d_src)

    cl.enqueue_copy(queue=queue, dest=h_dst, src=d_dst)
    cl.enqueue_copy(queue=queue, dest=h_src, src=d_src)
    queue.finish()

    print(h_src)
    print(h_dst)
    print(sum(h_src))


if __name__ == "__main__" :	
    #cp.run("main()")

    for i in range(1) :
        main()
