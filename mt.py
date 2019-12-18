import cProfile as cp
import logging as Log
import numpy as np
import os
import pyopencl as cl
import pyopencl.array as cl_array
import scan
import simpleOBJ as obj
import volume

class MarchingTet:
    def __init__(self, device, ctx, queue) :
        self.ctx = ctx 
        self.queue = queue
        with open('kernel/mt.cl', 'r') as fp : src = fp.read()
        if "NVIDIA" == device.get_info(cl.device_info.VENDOR)[:6] :
            self.prg = cl.Program(self.ctx, src).build(options=["-cl-nv-verbose"], devices=[device,])
            print(self.prg.get_build_info(device=device, param=cl.program_build_info.LOG))
        else :
            self.prg = cl.Program(self.ctx, src).build(devices=[device,])
        
        self.cscan = scan.Scan(self.ctx, self.queue)
        self.escan = scan.Scan(self.ctx, self.queue)
        self.nr_idx = np.zeros(1, dtype=np.int32)
        self.nr_vtx = np.zeros(1, dtype=np.int32)
        self.sz_idx = np.zeros(1, dtype=np.int32)
        self.sz_vtx = np.zeros(1, dtype=np.int32)
        self.evts = dict()
        self.mems = dict()

    def loadVolumeFromGPUMem(self, mem, dim) :
        # Volume
        self.dim = dim
        self.d_Volume = mem

        # Memory Size
        self.bcc_dim = np.int32((self.dim[0]//2,self.dim[1]//2,self.dim[2]//2,1))
        self.sz_lattice = self.bcc_dim[0]*self.bcc_dim[1]*self.bcc_dim[2]*2
        self.sz_ctype = self.bcc_dim[0]*self.bcc_dim[1]*self.bcc_dim[2]*2*6
        self.sz_etype = self.bcc_dim[0]*self.bcc_dim[1]*self.bcc_dim[2]*2*7
        self.sz_vtx = self.sz_ctype//2
        self.sz_idx = self.sz_ctype//2

        # Temporary Tables
        mf = cl.mem_flags
        self.d_Table = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_etype*4)
        self.d_Counter = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_etype*4)

        # Output        
        self.d_Index = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_idx*4)
        self.d_Vertex = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_vtx*4)
        self.d_Normal = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_vtx*4)

        self.cscan.initMem(self.sz_ctype)
        self.escan.initMem(self.sz_etype)

    def loadVolume(self, filename, dim) :
        self.dim = dim

        # Volume
        self.h_Volume = np.reshape(np.fromfile(filename, dtype=np.float32), self.dim)
        mf = cl.mem_flags
        fformat = cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT)
        self.d_Volume = cl.Image(self.ctx, mf.READ_ONLY, format=fformat, shape=self.dim)
        cl.enqueue_copy(queue=self.queue, dest=self.d_Volume, src=self.h_Volume, origin=(0,0,0), region=self.dim, is_blocking=True)
        # Temporary Tables
        self.bcc_dim = np.int32((self.dim[0]//2,self.dim[1]//2,self.dim[2]//2,1))
        self.sz_lattice = self.bcc_dim[0]*self.bcc_dim[1]*self.bcc_dim[2]*2
        self.sz_ctype = self.bcc_dim[0]*self.bcc_dim[1]*self.bcc_dim[2]*2*6
        self.sz_etype = self.bcc_dim[0]*self.bcc_dim[1]*self.bcc_dim[2]*2*7
        self.sz_vtx = self.sz_ctype//2
        self.sz_idx = self.sz_ctype//2
        
        self.d_Table = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_etype*4)
        self.d_Counter = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_etype*4)

        # Output        
        self.d_Index = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_idx*4)
        self.d_Vertex = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_vtx*4)
        self.d_Normal = cl.Buffer(self.ctx, mf.READ_WRITE, self.sz_vtx*4)

        self.cscan.initMem(self.sz_ctype)
        self.escan.initMem(self.sz_etype)

    def CellTest(self) :
        evt = dict()
        evt["Cell Test"] = self.prg.CellTest(self.queue, (self.sz_lattice, ), None, self.d_Counter, self.d_Table, self.d_Volume, self.isovalue, self.bcc_dim)
        evt["CellTable Scan"] = self.cscan.run(self.d_Counter, self.d_Counter)
        evt["CellTable Compaction"] = self.prg.ICompact(self.queue, (self.sz_ctype, ), None, self.d_Index, self.d_Counter, self.d_Table)
        evt["Read Index Count"] = cl.enqueue_copy(self.queue, self.nr_idx, self.d_Counter, device_offset=(self.sz_ctype-1)*4)
        self.evts["Cell Test"] = evt;
        self.sz_idx = (self.nr_idx+127)//128*128

    def EdgeTest(self) :
        evt = dict()
        evt["Edge Test"] = self.prg.EdgeTest(self.queue, (self.sz_lattice, ), None, self.d_Table, self.d_Volume, self.isovalue, self.bcc_dim)
        evt["EdgeTable Scan"] = self.escan.run(self.d_Counter, self.d_Table)
        evt["EdgeTable Compaction"] = self.prg.VCompact(self.queue, (self.sz_etype, ), None, self.d_Vertex, self.d_Table, self.d_Counter)
        evt["Read Vertex Count"] = cl.enqueue_copy(self.queue, self.nr_vtx, self.d_Counter, device_offset=(self.sz_etype-1)*4)
        self.evts["Edge Test"] = evt;
        self.sz_vtx = (self.nr_vtx+127)//128*128

    def GenVertex(self) :
        if self.nr_vtx[0]<1 : return
        evt = dict()
        evt["Vertex Extraction"] = self.prg.GenVertex(self.queue, (self.sz_vtx, ), None, self.d_Vertex, self.d_Volume, self.d_Vertex, self.isovalue, self.bcc_dim, self.nr_vtx)
        evt["Normal Estimation"] = self.prg.GenNormal(self.queue, (self.sz_vtx,), None, self.d_Normal, self.d_Volume, self.d_Vertex, self.bcc_dim, self.nr_vtx)
        self.evts["Vertex Processing"] = evt

    def GenIndex(self) : 
        if self.nr_idx[0]<1 : return
        evt = dict()
        evt["Index Generation"] = self.prg.GenIndex(self.queue, (self.sz_idx, ), None, 
            self.d_Index, self.d_Volume, self.d_Counter, np.int32(self.bcc_dim), self.isovalue, self.nr_idx)
        self.evts["Index Processing"] = evt

    def getVertices(self) :
        if self.nr_vtx[0]<1 : return None
        h_vtx = np.zeros((self.nr_vtx[0], 4), dtype=np.float32)
        cl.enqueue_copy(queue=self.queue, dest=h_vtx, src=self.d_Vertex, is_blocking=True)
        return h_vtx

    def getIndices(self) :
        if self.nr_idx[0]<1 : return None
        h_idx = np.zeros((self.nr_idx[0], 4), dtype=np.int32)
        cl.enqueue_copy(queue=self.queue, dest=h_idx, src=self.d_Index, is_blocking=True)
        return h_idx

    def getNormals(self) :
        if self.nr_vtx[0]<1 : return None 
        h_nor = np.zeros((self.nr_vtx[0], 4)).astype(np.float32)
        cl.enqueue_copy(queue=self.queue, dest=h_nor, src=self.d_Normal, is_blocking=True)
        return h_nor

    def saveAs(self, filename):
        Log.info("Save")
        Log.info("\t{0}".format(filename))
        h_vtx = self.getVertices()
        h_idx = self.getIndices()
        h_nor = self.getNormals()
        h_vtx = h_vtx / (np.max(self.dim)-1)*2-1
        h_vtx[:,3] = 1
        Log.info("\t# of vertices : {0}".format(h_vtx.shape))
        Log.info("\t# of indices : {0}".format(h_idx.shape))
        obj.saveOBJ(filename+".obj", h_vtx, h_nor, h_idx)

    def Profile(self) :
        Log.info("Performance")
        msec = (lambda evt:(evt.profile.end-evt.profile.start)*1E-6)
        total = 0

        self.queue.finish()
        for k,v in self.evts.items() :
            item = dict()
            for kk,vv in v.items() : item[kk] = sum([msec(x) for x in vv]) if isinstance(vv, list) else msec(vv)
            stage_tot = sum(item.values())
            Log.info("{0:s} : {1:.4f} msec".format(k, stage_tot))
            for kk, vv in item.items() : Log.info("  {0:s} : {1:.4f} msec".format(kk, vv))
            total = total + stage_tot
        Log.info("\tTotal : {0:.4f} msec( = {1:.4f} FPS)\n".format(total, (1000.0/total)))

        
    def isosurface(self, isovalue) :
        self.isovalue = np.float32(isovalue)
        self.CellTest()
        print("Cell Test Complete")
        self.EdgeTest()
        print("Edge Test Complete")
        self.GenVertex()
        print("Gen Vertex Complete")
        self.GenIndex()
        print("Gen Index Compete")
        self.queue.finish()
        print("Queue Finish")

def main() :
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    res = 64
    data = ( "dragon_vrip", (res*2, res*2, res) )
    
    mt = MarchingTet(devices[0],ctx,queue)
    mt.loadVolume("./data/{0}_FLT32_{1}_{2}_{3}.raw".format(data[0], *data[1]), data[1])
    mt.isosurface(0.0)
    #print("isosurface extraction complete")
    mt.Profile()

    mt.saveAs("./output/"+data[0])
    
if __name__ == "__main__" :    
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    fmt = Log.Formatter(fmt='[%(levelname)s][%(asctime)s][%(funcName)s:%(lineno)d] %(message)s', datefmt='%y/%m/%d-%H:%M:%S')
    hStream = Log.StreamHandler()
    hStream.setFormatter(fmt)
    hFile = Log.FileHandler('./output/basisRefinement.log')
    hFile.setFormatter(fmt)
    Log.basicConfig(format=fmt, level=Log.INFO, handlers=[hFile, hStream])
    #cp.run('main()')
    main()

