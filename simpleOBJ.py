import numpy as np
def loadOBJ(filename):
	with open(filename, "r") as fp :
		data = fp.read()
		fp.close()

	lines = data.split("\n")

	V = list()
	N = list()
	Fv = list()
	Ft = list()
	Fn = list()
	for line in lines :
		line = line.split(" ")
		
		while '' in line : line.remove('')
		if len(line)>0 :
			if line[0] == 'v' :
				V.append([float(x) for x in line[1:]])
			if line[0] == 'vn' :
				N.append([float(x) for x in line[1:]])
			elif line[0] == 'f' :
				fv = list()
				fn = list()
				ft = list()
				for l in line[1:] :
					ls = l.split('/')
					if len(ls) > 0 and ls[0] != '' : fv.append(int(ls[0])-1)
					if len(ls) > 1 and ls[1] != '' : ft.append(int(ls[1]))
					if len(ls) > 2 and ls[2] != '' : fn.append(int(ls[2]))
				Fv.append(fv)
				Fn.append(fn)
				Ft.append(ft)
	return {'vertex' : np.array(V).astype(np.float32), 'face' : np.array(Fv).astype(np.int32), 'normal' : np.array(N).astype(np.float32), 'faceNormal' : Fn}


def saveOBJ(filename, vtx, nor, idx):
	with open(filename, 'w') as fp :
		for v in vtx :
			fp.write("v {0} {1} {2}\n".format(v[0], v[1], v[2]))
		for n in nor :
			fp.write("vn {0} {1} {2}\n".format(n[0], n[1], n[2]))
		for f in idx :
			#if f[0]>0 and f[1]>0 and f[2]>0 :
			fp.write("f {0} {1} {2}\n".format(f[0]+1, f[1]+1, f[2]+1))



if __name__ == "__main__" :	
    obj = loadOBJ("/Users/kamu/Work/MarchingTetra/py_mt/test.obj")
