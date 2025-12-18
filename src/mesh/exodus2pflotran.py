import h5py
import numpy as np
from h5py import * 
from netCDF4 import * 
import numpy 
import os
import sys
import time


# filepath = HDF file name, path = dataset directory in HDF file
class HDF5Reader:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_dataset(self, path):
        with h5py.File(self.filepath, 'r') as file:
            data = file[path][:]
        return data


class ChangeFormatSideset:
    def __init__(self, data):
        self.data = data

    def formatting(self):
        self.data = self.data.astype(str)
        self.data[:, 0] = 'Q'


class ASCIIWriterSideset:
    def __init__(self, data):
        self.data = data
        self.col_num = self.data.shape[0]

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            file.write(f'{self.col_num}\n')
            for row in self.data:
                file.write('  '.join(row) + '\n')


class ChangeFormatMaterial:
    def __init__(self, data):
        self.data = data
        self.materials = np.unique(self.data)

    def formatting(self):

        material_dic = {}

        for material in self.materials:
            material_indices = np.where(self.data == material)[0] + 1
            material_dic[f"material_{material}"] = material_indices.tolist()

        return material_dic


class ASCIIWriterMaterial:
    def __init__(self, data):
        self.data = data

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            for i in range(len(self.data)):
                file.write(f'{self.data[i]}\n')


def exodus_to_pflotran_mesh(infilename, outfilename):

    input_file_path = infilename
    output_file_path = outfilename

    start_time = time.time()

    if not os.path.exists(input_file_path):
        print("ERROR: Input file not found.\n")
        sys.exit(1)

    print("Infile: %s Outfile: %s" %(infilename, outfilename))

    # Open Exodus NetCDF file
    exofile=Dataset(infilename)

    # Open PFLOTRAN HDF5 Mesh file
    pflotranfile = File(output_file_path, mode='w')

    # Let's read some dimensions out of the Exodus file
    
    # Read number of vertices
    num_vert=len(exofile.dimensions['num_nodes'])
    # Read number of elements
    num_elem=len(exofile.dimensions['num_elem'])
    # Number of side sets
    if 'num_side_sets' in exofile.dimensions:
        num_sidesets=len(exofile.dimensions['num_side_sets'])
        sideset_exist=True
    else: sideset_exist=False
    # Number of material blocks
    num_blocks=len(exofile.dimensions['num_el_blk'])
    
    block_size=numpy.zeros((num_blocks, 2), int)

    mat_id = numpy.zeros((num_elem), int)

    print("Preparing coordinates")

    # Let's read in x,y,z coordinates and write to the HDF5 file
    h5dset = pflotranfile.create_dataset('Domain/Vertices', (num_vert, 3), 'f8')
    x = exofile.variables['coordx'][:]
    y = exofile.variables['coordy'][:]
    z = exofile.variables['coordz'][:]
    h5dset[:,0] = x
    h5dset[:,1] = y
    h5dset[:,2] = z

    print("Preparing Domain")

    # We will loop through all blocks and find number of elements 
    # and number of nodes per element
    counter=0
    for i in range(num_blocks):
        print("Working on block %d" %(i+1))
        varname='num_el_in_blk'+str(i+1)
        block_size[i,0]=len(exofile.dimensions[varname])
        varname='num_nod_per_el'+str(i+1)
        block_size[i,1]=len(exofile.dimensions[varname])
        if i == 0:
            cell_array = numpy.zeros((num_elem, (block_size[i,1]+1)), int)

        varname='connect'+str(i+1)
        element_type = exofile.variables[varname].elem_type
        if (element_type.startswith('HEX')):
            num_vert_per_elem = 8
        elif (element_type.startswith('WEDGE')):
            num_vert_per_elem = 6
        else:
            sys.exit('ERROR: Unknown element type: {}'.format(element_type))
                 
        block = exofile.variables[varname][:]
        
        for j in range(block_size[i,0]):
            mat_id[counter] = i+1
            cell_array[counter,0] = num_vert_per_elem 
            for k in range(num_vert_per_elem):
                cell_array[counter, k+1] = block[j,k]
            # Check for repetitions
            outlist = []
            for ele in cell_array[counter,1:len(cell_array[counter,:])]:
                if ele not in outlist:
                    outlist.append(ele)
            cell_array[counter,:] = 0 
            cell_array[counter,0] = len(outlist) 
            cell_array[counter,1:len(outlist)+1] = outlist 
            counter += 1


    totsize = numpy.shape(cell_array)

    print(totsize)
    print("Block 1 %d %d" %(block_size[0,0], block_size[0,1]))

    h5dset = pflotranfile.create_dataset('Domain/Cells', data=cell_array)

    print("Preparing Material Ids")

    # Write Material Ids
    int_array = numpy.arange(num_elem)
    int_array += 1
    h5dset = pflotranfile.create_dataset('Materials/Cell Ids', data=int_array)
    h5dset = pflotranfile.create_dataset('Materials/Material Ids', data=mat_id)

    # Write sidesets
    for i in range(num_sidesets):
        varname='elem_ss'+str(i+1)
        elem = exofile.variables[varname][:]
        varname='side_ss'+str(i+1)
        side = exofile.variables[varname][:]

        print(numpy.shape(elem))

        if elem.size != side.size:
            print("Inconsistent size in the data set: "+str(varname))
            sys.exit(0)
        
                
        sideset = numpy.zeros((side.size,5), int)
        # We will check which face of the element is at the boundary and
        # create our sidesets accordingly
        for j in range(len(elem)):
            sideset[j,0] = 4
            jelem = elem[j] -1
            if (cell_array[jelem,0] == 6):
                if side[j] == 1:
                    sideset[j,1] = cell_array[jelem,1]
                    sideset[j,2] = cell_array[jelem,2]
                    sideset[j,3] = cell_array[jelem,5]
                    sideset[j,4] = cell_array[jelem,4]
                elif side[j] == 2:
                    sideset[j,1] = cell_array[jelem,2]
                    sideset[j,2] = cell_array[jelem,3]
                    sideset[j,3] = cell_array[jelem,6]
                    sideset[j,4] = cell_array[jelem,5]
                elif side[j] == 3:
                    sideset[j,1] = cell_array[jelem,3]
                    sideset[j,2] = cell_array[jelem,1]
                    sideset[j,3] = cell_array[jelem,4]
                    sideset[j,4] = cell_array[jelem,6]
                elif side[j] == 4:
                    sideset[j,0] = 3
                    sideset[j,1] = cell_array[jelem,3]
                    sideset[j,2] = cell_array[jelem,2]
                    sideset[j,3] = cell_array[jelem,1]
                elif side[j] == 5:
                    sideset[j,0] = 3
                    sideset[j,1] = cell_array[jelem,4]
                    sideset[j,2] = cell_array[jelem,5]
                    sideset[j,3] = cell_array[jelem,6]
            elif (cell_array[jelem,0] == 8):
                if side[j] == 1:
                    sideset[j,1] = cell_array[jelem,1]
                    sideset[j,2] = cell_array[jelem,2]
                    sideset[j,3] = cell_array[jelem,6]
                    sideset[j,4] = cell_array[jelem,5]
                elif side[j] == 2:
                    sideset[j,1] = cell_array[jelem,2]
                    sideset[j,2] = cell_array[jelem,3]
                    sideset[j,3] = cell_array[jelem,7]
                    sideset[j,4] = cell_array[jelem,6]
                elif side[j] == 3:
                    sideset[j,1] = cell_array[jelem,3]
                    sideset[j,2] = cell_array[jelem,7]
                    sideset[j,3] = cell_array[jelem,8]
                    sideset[j,4] = cell_array[jelem,4]
                elif side[j] == 4:
                    sideset[j,1] = cell_array[jelem,4]
                    sideset[j,2] = cell_array[jelem,8]
                    sideset[j,3] = cell_array[jelem,5]
                    sideset[j,4] = cell_array[jelem,1]
                elif side[j] == 5:
                    sideset[j,1] = cell_array[jelem,1]
                    sideset[j,2] = cell_array[jelem,2]
                    sideset[j,3] = cell_array[jelem,3]
                    sideset[j,4] = cell_array[jelem,4]
                elif side[j] == 6:
                    sideset[j,1] = cell_array[jelem,5]
                    sideset[j,2] = cell_array[jelem,6]
                    sideset[j,3] = cell_array[jelem,7]
                    sideset[j,4] = cell_array[jelem,8]


        dataset_name = 'Regions/Sideset%d' % (i+1)
        h5dset = pflotranfile.create_dataset(dataset_name, data=sideset)

    return sideset_exist


if __name__ == "__main__":
    
    exodus_path = './src/mesh/mesh_128.e'
    hdf_path = './src/mesh/output_128/mesh.h5'
    sideset_dataset_path = ['/Regions/Sideset1', '/Regions/Sideset2']
    ascii_file_path = ['./src/mesh/output_128/sideset1.ss', './src/mesh/output_128/sideset2.ss']
    material_dataset_path = '/Materials/Material Ids'
    
    exodus_to_pflotran_mesh(exodus_path, hdf_path)

    reader = HDF5Reader(hdf_path)

    for i in range(len(sideset_dataset_path)):
        data = reader.read_dataset(sideset_dataset_path[i])
        processor = ChangeFormatSideset(data)
        processor.formatting()
        writer = ASCIIWriterSideset(processor.data)
        writer.save_to_file(ascii_file_path[i])

    data = reader.read_dataset(material_dataset_path)
    processor = ChangeFormatMaterial(data)
    material_dic = processor.formatting()

    for key in material_dic.keys():
        writer = ASCIIWriterMaterial(material_dic[key])
        writer.save_to_file(f'./src/mesh/output_128/{key}.txt')

