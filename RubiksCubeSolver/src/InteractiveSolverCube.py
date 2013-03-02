#----------------------------------------------------------------------
# Rubiks cube solver
# Written by Brendan Hickey
# Visualizer adapted from cube code written by David Hogg and Jake Vanderplas
#   https://github.com/davidwhogg/MagicCube



"""
This program shows a representation of a rubiks cube that users can scramble
and provides options to solve the cube, "quick solve" or "hard solve". 
The hard solve option looks to solve the cube without using any predetermined 
algorithms. This is highly inefficient and slow, but provides an interesting 
programming challenge as a rubik's cube has 43x10^18 states, and no easy way
to know number of moves from current state to solved state. 

The program could be significantly speeded up by using predetermined algorithms 
for common moves such as cube twist - essentially mimicking how humans solve 
the cube. This program is purely a programming exercise, and the goal is to 
solve a cube without using any predetermined algorithms on a laptop computer, 
so predetermined algorithms are not used.  

The solve works in two phases. There is a subset of cube states (CS') that you
can reach from solved state by twisting only top, left, right faces. The 
program takes the scrambled cube and first moves it to CS' using all sides. 
In a second phase, it then limits moves to T,L,R faces to reach the solution. 
The search itself is a breadth first search. 

Cube scoring is done by counting whether edges (2 sticker cubes) are in solved
or easily solvable state, and whether corner cubes are in solved state. By 
applying weights, the cube is given a score. 


Speed Improvements
------------------
Once in CS', there are still a significant number of moves needed to get to 
solved state. The time taken is improved by creation of a dictionary containing
cube states that can be reached from solved state using T,L,R. Duplicate cube 
states are only represented once using the shortest path to that state. Every
time a move is executed after phase 1, the dictionary is checked to see if we 
have that state.

The tree pruning was also done by eliminating two subsequent moves of same 
face, and also eliminating two moves sequences that are alternatively covered
- e.g. left twist then right twist is covered by right then left.


"""


"""
There are two cube representations. Stickers provides physical
locations in 3d space. Cube_string represents the cube configuration. 


Cube String
-----------
Face sides are Top,Left,Front,Right,Back and Under (T,L,F,R,B,U)
For a particular cube, each side corresponds 1:1 to a color of center sticker. 
A sticker color can be therefore represented by a char t,l,f,r,b,u
Starting at top left and going left to right a side can be represented by a string
e.g. 'ttttttttt' for a solved side or 'ttlltbuur' for a scrambled side. 
Concatenating the faces together in order gives a string
'tttttttttlllllllllfffffffffrrrrrrrrrbbbbbbbbbuuuuuuuuu' for a solved cube or
'ltbftufbbblrblrllutrrtfblfbultlrrlrrrftubtubffuutuftuf' for a scrambled cube


Sticker representation
----------------------
Each face is represented by a length [5, 3] array:

  [v1, v2, v3, v4, v1]

Each sticker is represented by a length [9, 3] array:

  [v1a, v1b, v2a, v2b, v3a, v3b, v4a, v4b, v1a]

In both cases, the first point is repeated to close the polygon.

Each face also has a centroid, with the face number appended
at the end in order to sort correctly using lexsort.
The centroid is equal to sum_i[vi].

Colors are accounted for using color indices and a look-up table.

With all faces in an NxNxN cube, then, we have three arrays:

  centroids.shape = (6 * N * N, 4)
  faces.shape = (6 * N * N, 5, 3)
  stickers.shape = (6 * N * N, 9, 3)
  colors.shape = (6 * N * N,)

The canonical order is found by doing

  ind = np.lexsort(centroids.T)

After any rotation, this can be used to quickly restore the cube to
canonical position.
"""


"""

Improvements to make:

1. Need to better integrate functions into classes. Some are duplicated. 
2. Come up with better "compress cube" function. 
3. Have dictionary load just once. Should be fixed with (1.)
4. Better order moves to speed solution for difficult cases. 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from projection import Quaternion, project_points

import time
import sys, os
import collections
import cPickle
#import random,string


class Cube:
    """Magic Cube Representation"""
    # define some attribues
    default_plastic_color = 'black'
    default_face_colors = ["w", "#ffcf00",
                           "#00008f", "#009f0f",
                           "#ff6f00", "#cf0000",
                           "gray", "none"]
    base_face = np.array([[1, 1, 1],
                          [1, -1, 1],
                          [-1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]], dtype=float)
    stickerwidth = 0.9
    stickermargin = 0.5 * (1. - stickerwidth)
    stickerthickness = 0.001
    (d1, d2, d3) = (1 - stickermargin,
                    1 - 2 * stickermargin,
                    1 + stickerthickness)
    base_sticker = np.array([[d1, d2, d3], [d2, d1, d3],
                             [-d2, d1, d3], [-d1, d2, d3],
                             [-d1, -d2, d3], [-d2, -d1, d3],
                             [d2, -d1, d3], [d1, -d2, d3],
                             [d1, d2, d3]], dtype=float)

    base_face_centroid = np.array([[0, 0, 1]])
    base_sticker_centroid = np.array([[0, 0, 1 + stickerthickness]])

    # Define rotation angles and axes for the six sides of the cube
    x, y, z = np.eye(3)
    rots = [Quaternion.from_v_theta(x, theta)
            for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(y, theta)
             for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

    # define face movements
    facesdict = dict(F=z, B=-z,
                     R=x, L=-x,
                     U=y, D=-y)

    def __init__(self, N=3, plastic_color=None, face_colors=None,colors=[]):
        self.N = N
        self.colors=colors
        if plastic_color is None:
            self.plastic_color = self.default_plastic_color
        else:
            self.plastic_color = plastic_color

        if face_colors is None:
            self.face_colors = self.default_face_colors
        else:
            self.face_colors = face_colors
          
        #Cube starts as solved.   
        self.cube_string=\
        'tttttttttlllllllllfffffffffrrrrrrrrrbbbbbbbbbuuuuuuuuu'

        self._move_list = []
        self._initialize_arrays()

    def _initialize_arrays(self):
        # initialize centroids, faces, and stickers.  We start with a
        # base for each one, and then translate & rotate them into position.

        # Define N^2 translations for each face of the cube
        cubie_width = 2. / self.N
        translations = np.array([[[-1 + (i + 0.5) * cubie_width,
                                   -1 + (j + 0.5) * cubie_width, 0]]
                                 for i in range(self.N)
                                 for j in range(self.N)])

        # Create arrays for centroids, faces, stickers, and colors
        face_centroids = []
        faces = []
        sticker_centroids = []
        stickers = []
        #colors = []

        factor = np.array([1. / self.N, 1. / self.N, 1])

        for i in range(6):
            M = self.rots[i].as_rotation_matrix()
            faces_t = np.dot(factor * self.base_face
                             + translations, M.T)
            stickers_t = np.dot(factor * self.base_sticker
                                + translations, M.T)
            face_centroids_t = np.dot(self.base_face_centroid
                                      + translations, M.T)
            sticker_centroids_t = np.dot(self.base_sticker_centroid
                                         + translations, M.T)
            
            colors_i = i + np.zeros(face_centroids_t.shape[0], dtype=int)

            # append face ID to the face centroids for lex-sorting
            face_centroids_t = np.hstack([face_centroids_t.reshape(-1, 3),
                                          colors_i[:, None]])
            sticker_centroids_t = sticker_centroids_t.reshape((-1, 3))

            faces.append(faces_t)
            face_centroids.append(face_centroids_t)
            stickers.append(stickers_t)
            sticker_centroids.append(sticker_centroids_t)            
            #self.colors.append(colors_i)
            
              
        self._face_centroids = np.vstack(face_centroids)
        self._faces = np.vstack(faces)
        self._sticker_centroids = np.vstack(sticker_centroids)
        self._stickers = np.vstack(stickers)
        self._colors = np.concatenate(self.colors)

        self._sort_faces()

    def _sort_faces(self):
        # use lexsort on the centroids to put faces in a standard order.
        ind = np.lexsort(self._face_centroids.T)
        self._face_centroids = self._face_centroids[ind]
        self._sticker_centroids = self._sticker_centroids[ind]
        self._stickers = self._stickers[ind]
        self._colors = self._colors[ind]
        self._faces = self._faces[ind]

    def rotate_face(self, f, n=1, layer=0,turns=1,move_start=True):
        """Rotate Face                               """
        """f is char denoting face - F,R,U,B,L,D     """
        """n is number of 1/4 turns with +- direction """
        
        if move_start:
            move_convert={'U1':'A','U2':'B','U3':'C','L1':'D','L2':'E','L3':'F',\
                          'F1':'G','F2':'H','F3':'I','R1':'J','R2':'K','R3':'L',\
                          'B1':'M','B2':'N','B3':'O','D1':'P','D2':'Q','D3':'R'}
            
            turns_int=int(round(turns))
            if turns_int<0: clockw_t=4+turns_int
            else: clockw_t=turns_int
            
            string_move=move_convert[f+str(clockw_t)]
            
            self.cube_string=cube_side_move(self.cube_string,string_move)
            #print "Cube is",self.cube_string," string is",string_move,\
            #    "turns is",turns,"face is ",f
        
        if layer < 0 or layer >= self.N:
            raise ValueError('layer should be between 0 and N-1')

        try:
            f_last, n_last, layer_last = self._move_list[-1]
        except:
            f_last, n_last, layer_last = None, None, None

        if (f == f_last) and (layer == layer_last):
            ntot = (n_last + n) % 4
            if abs(ntot - 4) < abs(ntot):
                ntot = ntot - 4
            if np.allclose(ntot, 0):
                self._move_list = self._move_list[:-1]
            else:
                self._move_list[-1] = (f, ntot, layer)
        else:
            self._move_list.append((f, n, layer))
        
        v = self.facesdict[f]        
        r = Quaternion.from_v_theta(v, n * np.pi / 2)
        M = r.as_rotation_matrix()

        proj = np.dot(self._face_centroids[:, :3], v)
        cubie_width = 2. / self.N
        flag = ((proj > 0.9 - (layer + 1) * cubie_width) &
                (proj < 1.1 - layer * cubie_width))

        for x in [self._stickers, self._sticker_centroids,self._faces]:
            x[flag] = np.dot(x[flag], M.T)
        self._face_centroids[flag, :3] = np.dot(self._face_centroids[flag, :3],M.T)
        
    def cube_side_move(self,rcube,rcube_move):    
        
    #Cube Moves
    #----------
    #Moves are defined by which face, and a clockwise move 90, 180, or 270 degrees. 
    #Top90, Top180, Top270 is moves A, B and C respectively (Yellow)
    #Left90, Left180, Left270 is moves D, E and F respectively
    #Front90, Front180, Front270 is moves G, H and I respectively
    #Right90, Right180, Right270 is moves J, K and L respectively
    #Back90, Back180, Back270 is moves M, N and O respectively
    #Under90, Under180, Under270 is moves P, Q and R respectively

        rcube_new = list(rcube)
        if rcube_move=='A':  # 'T1':  
            rcube_new[0], rcube_new[1], rcube_new[2] =rcube[6], rcube[3], rcube[0]
            rcube_new[3], rcube_new[5]=rcube[7], rcube[1] #don't need to copy center
            rcube_new[6], rcube_new[7], rcube_new[8] =rcube[8], rcube[5], rcube[2] 
            rcube_new[9], rcube_new[10],rcube_new[11]=rcube[18],rcube[19],rcube[20]
            rcube_new[18],rcube_new[19],rcube_new[20]=rcube[27],rcube[28],rcube[29]   
            rcube_new[27],rcube_new[28],rcube_new[29]=rcube[36],rcube[37],rcube[38]  
            rcube_new[36],rcube_new[37],rcube_new[38]=rcube[9], rcube[10],rcube[11]
    
        elif rcube_move =='B': # 'T2':
            rcube_new[9], rcube_new[10],rcube_new[11]=rcube[27],rcube[28],rcube[29]
            rcube_new[18],rcube_new[19],rcube_new[20]=rcube[36],rcube[37],rcube[38]
            rcube_new[27],rcube_new[28],rcube_new[29]=rcube[9], rcube[10],rcube[11]  
            rcube_new[36],rcube_new[37],rcube_new[38]=rcube[18],rcube[19],rcube[20]
            rcube_new[0], rcube_new[1], rcube_new[2] =rcube[8], rcube[7], rcube[6]
            rcube_new[3],               rcube_new[5] =rcube[5],           rcube[3]        
            rcube_new[6], rcube_new[7], rcube_new[8] =rcube[2], rcube[1], rcube[0]
    
        elif rcube_move=='C':  # 'T3':
            rcube_new[9], rcube_new[10],rcube_new[11]=rcube[36],rcube[37],rcube[38]   
            rcube_new[18],rcube_new[19],rcube_new[20]=rcube[9], rcube[10],rcube[11]  
            rcube_new[27],rcube_new[28],rcube_new[29]=rcube[18],rcube[19],rcube[20]
            rcube_new[36],rcube_new[37],rcube_new[38]=rcube[27],rcube[28],rcube[29]
            rcube_new[0], rcube_new[1], rcube_new[2] =rcube[2], rcube[5], rcube[8]
            rcube_new[3],               rcube_new[5] =rcube[1],           rcube[7]        
            rcube_new[6], rcube_new[7], rcube_new[8] =rcube[0], rcube[3], rcube[6] 
    
        elif rcube_move=='D':  # 'L1':
            rcube_new[0], rcube_new[3], rcube_new[6] =rcube[44],rcube[41],rcube[38] 
            rcube_new[18],rcube_new[21],rcube_new[24]=rcube[0], rcube[3], rcube[6] 
            rcube_new[38],rcube_new[41],rcube_new[44]=rcube[51],rcube[48],rcube[45] 
            rcube_new[51],rcube_new[48],rcube_new[45]=rcube[24],rcube[21],rcube[18]
            rcube_new[9], rcube_new[10],rcube_new[11]=rcube[15],rcube[12],rcube[9]
            rcube_new[12],              rcube_new[14]=rcube[16],          rcube[10]      
            rcube_new[15],rcube_new[16],rcube_new[17]=rcube[17],rcube[14],rcube[11]
    
        elif rcube_move=='E': # 'L2':
            rcube_new[0], rcube_new[3], rcube_new[6] =rcube[45],rcube[48],rcube[51]
            rcube_new[18],rcube_new[21],rcube_new[24]=rcube[44],rcube[41],rcube[38]   
            rcube_new[38],rcube_new[41],rcube_new[44]=rcube[24],rcube[21],rcube[18]
            rcube_new[51],rcube_new[48],rcube_new[45]=rcube[6], rcube[3], rcube[0]
            rcube_new[9], rcube_new[10],rcube_new[11]=rcube[17],rcube[16],rcube[15]
            rcube_new[12],              rcube_new[14]=rcube[14],          rcube[12]      
            rcube_new[15],rcube_new[16],rcube_new[17]=rcube[11],rcube[10],rcube[9]
    
        elif rcube_move=='F':  # 'L3':
            rcube_new[0], rcube_new[3], rcube_new[6] =rcube[18],rcube[21],rcube[24] 
            rcube_new[18],rcube_new[21],rcube_new[24]=rcube[45],rcube[48],rcube[51] 
            rcube_new[38],rcube_new[41],rcube_new[44]=rcube[6], rcube[3], rcube[0] 
            rcube_new[51],rcube_new[48],rcube_new[45]=rcube[38],rcube[41],rcube[44]
            rcube_new[9], rcube_new[10],rcube_new[11]=rcube[11],rcube[14],rcube[17]
            rcube_new[12],              rcube_new[14]=rcube[10],          rcube[16]       
            rcube_new[15],rcube_new[16],rcube_new[17]=rcube[9], rcube[12],rcube[15] 
                
        elif rcube_move=='G':  # 'F1':
            rcube_new[6], rcube_new[7], rcube_new[8] =rcube[17],rcube[14],rcube[11]   
            rcube_new[11],rcube_new[14],rcube_new[17]=rcube[45],rcube[46],rcube[47]  
            rcube_new[27],rcube_new[30],rcube_new[33]=rcube[6], rcube[7], rcube[8] 
            rcube_new[45],rcube_new[46],rcube_new[47]=rcube[33],rcube[30],rcube[27] 
            rcube_new[18],rcube_new[19],rcube_new[20]=rcube[24],rcube[21],rcube[18]
            rcube_new[21],              rcube_new[23]=rcube[25] ,         rcube[19]       
            rcube_new[24],rcube_new[25],rcube_new[26]=rcube[26],rcube[23],rcube[20] 
          
        elif rcube_move=='H':   # 'F2':
            rcube_new[6], rcube_new[7], rcube_new[8] =rcube[47],rcube[46],rcube[45]
            rcube_new[11],rcube_new[14],rcube_new[17]=rcube[33],rcube[30],rcube[27] 
            rcube_new[27],rcube_new[30],rcube_new[33]=rcube[17],rcube[14],rcube[11]  
            rcube_new[45],rcube_new[46],rcube_new[47]=rcube[8], rcube[7], rcube[6] 
            rcube_new[18],rcube_new[19],rcube_new[20]=rcube[26],rcube[25],rcube[24]
            rcube_new[21],              rcube_new[23]=rcube[23],          rcube[21]        
            rcube_new[24],rcube_new[25],rcube_new[26]=rcube[20],rcube[19],rcube[18]
                     
        elif rcube_move=='I':  # 'F3':
            rcube_new[6], rcube_new[7], rcube_new[8] =rcube[27],rcube[30],rcube[33] 
            rcube_new[11],rcube_new[14],rcube_new[17]=rcube[8], rcube[7], rcube[6] 
            rcube_new[27],rcube_new[30],rcube_new[33]=rcube[47],rcube[46],rcube[45]    
            rcube_new[45],rcube_new[46],rcube_new[47]=rcube[11],rcube[14],rcube[17] 
            rcube_new[18],rcube_new[19],rcube_new[20]=rcube[20],rcube[23],rcube[26]
            rcube_new[21],              rcube_new[23]=rcube[19],          rcube[25]        
            rcube_new[24],rcube_new[25],rcube_new[26]=rcube[18],rcube[21],rcube[24]
    
        elif rcube_move=='J':  # 'R1':
            rcube_new[8], rcube_new[5], rcube_new[2] =rcube[26],rcube[23],rcube[20] 
            rcube_new[26],rcube_new[23],rcube_new[20]=rcube[53],rcube[50],rcube[47]  
            rcube_new[36],rcube_new[39],rcube_new[42]=rcube[8], rcube[5], rcube[2]
            rcube_new[47],rcube_new[50],rcube_new[53]=rcube[42],rcube[39],rcube[36]
            rcube_new[27],rcube_new[28],rcube_new[29]=rcube[33],rcube[30],rcube[27]
            rcube_new[30],              rcube_new[32]=rcube[34],          rcube[28]        
            rcube_new[33],rcube_new[34],rcube_new[35]=rcube[35],rcube[32],rcube[29]
    
        elif rcube_move=='K': # 'R2':
            rcube_new[8], rcube_new[5], rcube_new[2] =rcube[53],rcube[50],rcube[47] 
            rcube_new[26],rcube_new[23],rcube_new[20]=rcube[36],rcube[39],rcube[42] 
            rcube_new[36],rcube_new[39],rcube_new[42]=rcube[26],rcube[23],rcube[20] 
            rcube_new[47],rcube_new[50],rcube_new[53]=rcube[2], rcube[5], rcube[8]
            rcube_new[27],rcube_new[28],rcube_new[29]=rcube[35],rcube[34],rcube[33]
            rcube_new[30],              rcube_new[32]=rcube[32],rcube[30]       
            rcube_new[33],rcube_new[34],rcube_new[35]=rcube[29],rcube[28],rcube[27] 
    
        elif rcube_move=='L':   # 'R3':
            rcube_new[8], rcube_new[5], rcube_new[2] =rcube[36],rcube[39],rcube[42]
            rcube_new[26],rcube_new[23],rcube_new[20]=rcube[8], rcube[5], rcube[2] 
            rcube_new[36],rcube_new[39],rcube_new[42]=rcube[53],rcube[50],rcube[47] 
            rcube_new[47],rcube_new[50],rcube_new[53]=rcube[20],rcube[23],rcube[26]
            rcube_new[27],rcube_new[28],rcube_new[29]=rcube[29],rcube[32],rcube[35]
            rcube_new[30],              rcube_new[32]=rcube[28],          rcube[34]      
            rcube_new[33],rcube_new[34],rcube_new[35]=rcube[27],rcube[30],rcube[33] 
    
        elif rcube_move=='M':   # 'B1':
            rcube_new[2], rcube_new[1], rcube_new[0] =rcube[35],rcube[32],rcube[29]
            rcube_new[9], rcube_new[12],rcube_new[15]=rcube[2], rcube[1], rcube[0]
            rcube_new[29],rcube_new[32],rcube_new[35]=rcube[53],rcube[52],rcube[51]
            rcube_new[53],rcube_new[52],rcube_new[51]=rcube[15],rcube[12],rcube[9]
            rcube_new[36],rcube_new[37],rcube_new[38]=rcube[42],rcube[39],rcube[36]
            rcube_new[39],              rcube_new[41]=rcube[43],          rcube[37]
            rcube_new[42],rcube_new[43],rcube_new[44] =rcube[44],rcube[41],rcube[38]
            
        elif rcube_move=='N':  # 'B2':
            rcube_new[2], rcube_new[1], rcube_new[0] =rcube[51],rcube[52],rcube[53] 
            rcube_new[9], rcube_new[12],rcube_new[15]=rcube[35],rcube[32],rcube[29] 
            rcube_new[29],rcube_new[32],rcube_new[35]=rcube[15],rcube[12],rcube[9]
            rcube_new[53],rcube_new[52],rcube_new[51]=rcube[0], rcube[1], rcube[2]
            rcube_new[36],rcube_new[37],rcube_new[38]=rcube[44],rcube[43],rcube[42]
            rcube_new[39],              rcube_new[41]=rcube[41],          rcube[39]       
            rcube_new[42],rcube_new[43],rcube_new[44]=rcube[38],rcube[37],rcube[36] 
    
        elif rcube_move=='O':  # 'B3':
            rcube_new[2], rcube_new[1], rcube_new[0] =rcube[9], rcube[12],rcube[15] 
            rcube_new[9], rcube_new[12],rcube_new[15]=rcube[51],rcube[52],rcube[53]
            rcube_new[29],rcube_new[32],rcube_new[35]=rcube[0], rcube[1], rcube[2]
            rcube_new[53],rcube_new[52],rcube_new[51]=rcube[29],rcube[32],rcube[35]
            rcube_new[36],rcube_new[37],rcube_new[38]=rcube[38],rcube[41],rcube[44]
            rcube_new[39],              rcube_new[41]=rcube[37],          rcube[43]       
            rcube_new[42],rcube_new[43],rcube_new[44]=rcube[36],rcube[39],rcube[42]  
            
        elif rcube_move=='P': # 'U1':
            rcube_new[15],rcube_new[16],rcube_new[17]=rcube[42],rcube[43],rcube[44] 
            rcube_new[24],rcube_new[25],rcube_new[26]=rcube[15],rcube[16],rcube[17] 
            rcube_new[33],rcube_new[34],rcube_new[35]=rcube[24],rcube[25],rcube[26]              
            rcube_new[42],rcube_new[43],rcube_new[44]=rcube[33],rcube[34],rcube[35]
            rcube_new[45],rcube_new[46],rcube_new[47]=rcube[51],rcube[48],rcube[45]
            rcube_new[48],              rcube_new[50]=rcube[52],          rcube[46]       
            rcube_new[51],rcube_new[52],rcube_new[53]=rcube[53],rcube[50],rcube[47] 
    
        elif rcube_move=='Q':   # 'U2':
            rcube_new[15],rcube_new[16],rcube_new[17]=rcube[33],rcube[34],rcube[35] 
            rcube_new[24],rcube_new[25],rcube_new[26]=rcube[42],rcube[43],rcube[44]  
            rcube_new[33],rcube_new[34],rcube_new[35]=rcube[15],rcube[16],rcube[17]  
            rcube_new[42],rcube_new[43],rcube_new[44]=rcube[24],rcube[25],rcube[26] 
            rcube_new[45],rcube_new[46],rcube_new[47]=rcube[53],rcube[52],rcube[51]
            rcube_new[48],              rcube_new[50]=rcube[50],          rcube[48]     
            rcube_new[51],rcube_new[52],rcube_new[53]=rcube[47],rcube[46],rcube[45]
     
        elif rcube_move=='R':  # 'U3':
            rcube_new[15],rcube_new[16],rcube_new[17]=rcube[24],rcube[25],rcube[26] 
            rcube_new[24],rcube_new[25],rcube_new[26]=rcube[33],rcube[34],rcube[35]  
            rcube_new[33],rcube_new[34],rcube_new[35]=rcube[42],rcube[43],rcube[44]
            rcube_new[42],rcube_new[43],rcube_new[44]=rcube[15],rcube[16],rcube[17]
            rcube_new[45],rcube_new[46],rcube_new[47]=rcube[47],rcube[50],rcube[53]
            rcube_new[48],              rcube_new[50]=rcube[46],          rcube[52]       
            rcube_new[51],rcube_new[52],rcube_new[53]=rcube[45],rcube[48],rcube[51]
    
        return ''.join(rcube_new) 

    def draw_interactive(self):
        fig = plt.figure(figsize=(5, 5))
        fig.add_axes(InteractiveCube(self))
        return fig


class InteractiveCube(plt.Axes):
    def __init__(self, cube=None,
                 interactive=True,
                 view=(0, 0, 10),
                 fig=None, rect=[0, 0.16, 1, 0.84],
                 **kwargs):
        if cube is None:
            self.cube = Cube(3)
        elif isinstance(cube, Cube):
            self.cube = cube
        else:
            self.cube = Cube(cube)

        self._view = view
        self._start_rot = Quaternion.from_v_theta((1, -1, 0),
                                                  -np.pi / 6)

        if fig is None:
            fig = plt.gcf()

        # disable default key press events
        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        # add some defaults, and draw axes
        kwargs.update(dict(aspect=kwargs.get('aspect', 'equal'),
                           xlim=kwargs.get('xlim', (-2.0, 2.0)),
                           ylim=kwargs.get('ylim', (-2.0, 2.0)),
                           frameon=kwargs.get('frameon', False),
                           xticks=kwargs.get('xticks', []),
                           yticks=kwargs.get('yticks', [])))
        super(InteractiveCube, self).__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        self._start_xlim = kwargs['xlim']
        self._start_ylim = kwargs['ylim']

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        self._ax_LR_alt = (0, 0, 1)

        # Internal state variable
        self._active = False  # true when mouse is over axes
        self._button1 = False  # true when button 1 is pressed
        self._button2 = False  # true when button 2 is pressed
        self._event_xy = None  # store xy position of mouse event
        self._shift = False  # shift key pressed
        self._digit_flags = np.zeros(10, dtype=bool)  # digits 0-9 pressed

        self._current_rot = self._start_rot  #current rotation state
        self._face_polys = None
        self._sticker_polys = None

        self._draw_cube()

        # connect some GUI events
        self.figure.canvas.mpl_connect('button_press_event',
                                       self._mouse_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self._mouse_release)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self._mouse_motion)
        self.figure.canvas.mpl_connect('key_press_event',
                                       self._key_press)
        self.figure.canvas.mpl_connect('key_release_event',
                                       self._key_release)

        self._initialize_widgets()

        # write some instructions
        self.figure.text(0.01, 0.05,
                         "Mouse/arrow keys adjust view\n"
                         "U/D/L/R/B/F keys turn faces\n"
                         "(hold shift for counter-clockwise)",
                         size=10)

    def _initialize_widgets(self):
        self._ax_reset = self.figure.add_axes([0.79, 0.05, 0.18, 0.075])
        self._btn_reset = widgets.Button(self._ax_reset, 'Reset View')
        self._btn_reset.on_clicked(self._reset_view)

        self._ax_solve = self.figure.add_axes([0.61, 0.05, 0.18, 0.075])
        self._btn_solve = widgets.Button(self._ax_solve, 'Solve Cube')
        self._btn_solve.on_clicked(self._solve_cube)
        
        self._ax_hardsolve = self.figure.add_axes([0.43,0.05,0.18,0.075])
        self._btn_hardsolve = widgets.Button(self._ax_hardsolve,'Hard Solve')
        self._btn_hardsolve.on_clicked(self._hard_solve_cube)
        

    def _project(self, pts):
        return project_points(pts, self._current_rot, self._view, [0, 1, 0])

    def _draw_cube(self):
        stickers = self._project(self.cube._stickers)[:, :, :2]
        faces = self._project(self.cube._faces)[:, :, :2]
        face_centroids = self._project(self.cube._face_centroids[:, :3])
        sticker_centroids = self._project(self.cube._sticker_centroids[:, :3])

        plastic_color = self.cube.plastic_color
        colors = np.asarray(self.cube.face_colors)[self.cube._colors]   
        face_zorders = -face_centroids[:, 2]
        sticker_zorders = -sticker_centroids[:, 2]
        

        if self._face_polys is None:
            # initial call: create polygon objects and add to axes
            self._face_polys = []
            self._sticker_polys = []

            for i in range(len(colors)):
                fp = plt.Polygon(faces[i], facecolor=plastic_color,
                                 zorder=face_zorders[i])
                sp = plt.Polygon(stickers[i], facecolor=colors[i],
                                 zorder=sticker_zorders[i])

                self._face_polys.append(fp)
                self._sticker_polys.append(sp)
                self.add_patch(fp)
                self.add_patch(sp)
        else:
            # subsequent call: update the polygon objects
          
            for i in range(len(colors)):
                self._face_polys[i].set_xy(faces[i])
                self._face_polys[i].set_zorder(face_zorders[i])
                self._face_polys[i].set_facecolor(plastic_color)
            
                self._sticker_polys[i].set_xy(stickers[i])
                self._sticker_polys[i].set_zorder(sticker_zorders[i])
                self._sticker_polys[i].set_facecolor(colors[i])

        self.figure.canvas.draw()

    def rotate(self, rot):
        self._current_rot = self._current_rot * rot

    def rotate_face(self, face, turns=1, layer=0, steps=5):
        
        """Turns is # of 1/4 turns      """
        move_start=True
        if not np.allclose(turns, 0):
            for i in range(steps):
                self.cube.rotate_face(face, turns * 1. / steps,
                                      layer=layer,turns=turns,move_start=move_start)
                move_start=False
                self._draw_cube()

    def _reset_view(self, *args):
        self.set_xlim(self._start_xlim)
        self.set_ylim(self._start_ylim)
        self._current_rot = self._start_rot
        self._draw_cube()

    def _solve_cube(self, *args):
        move_list = self.cube._move_list[:]
        for (face, n, layer) in move_list[::-1]:
            self.rotate_face(face, -n, layer, steps=3)
        self.cube._move_list = []
        
    def _hard_solve_cube(self, *args):
        move_list = self.cube._move_list[:]
        print "Easy Move list length is ",len(move_list)
        
        solve_moves=solve_rubiks_cube(self.cube.cube_string)
        
        move_convert={'A':'U1','B':'U2','C':'U3','D':'L1','E':'L2','F':'L3',
                      'G':'F1','H':'F2','I':'F3','J':'R1','K':'R2','L':'R3',
                      'M':'B1','N':'B2','O':'B3','P':'D1','Q':'D2','R':'D3'}
        
        for mychar in solve_moves:
            temp_move=move_convert[mychar]
            face=temp_move[0]
            qturns=int(temp_move[1])
            if qturns<3: n=qturns
            else: n=-1 #3 cw q turns is 1 ccw qturn
            
            self.rotate_face(face, n, 0, steps=10*abs(n)) #steps should be 3
        self.cube._move_list = []

    def _key_press(self, event):
        """Handler for key press events"""
        
        #if event.key == 'shift':
        #    self._shift = True
        if event.key.isdigit():
            self._digit_flags[int(event.key)] = 1
        elif event.key == 'right':
            if self._shift:
                ax_LR = self._ax_LR_alt
            else:
                ax_LR = self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_LR,
                                                5 * self._step_LR))
        elif event.key == 'left':
            if self._shift:
                ax_LR = self._ax_LR_alt
            else:
                ax_LR = self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_LR,
                                                -5 * self._step_LR))
        elif event.key == 'up':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                5 * self._step_UD))
        elif event.key == 'down':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                -5 * self._step_UD))
        elif event.key.upper() in 'LRUDBF':
            if event.key in 'LRUDBF':   
                direction = -1
            else:
                direction = 1

            if np.any(self._digit_flags[:N]):
                for d in np.arange(N)[self._digit_flags[:N]]:
                    self.rotate_face(event.key.upper(), direction, layer=d)
            else:
                self.rotate_face(event.key.upper(), direction)
                
        self._draw_cube()

    def _key_release(self, event):
        """Handler for key release event"""
        if event.key == 'shift':
            self._shift = False
        elif event.key.isdigit():
            self._digit_flags[int(event.key)] = 0

    def _mouse_press(self, event):
        """Handler for mouse button press"""
        self._event_xy = (event.x, event.y)
        if event.button == 1:
            self._button1 = True
        elif event.button == 3:
            self._button2 = True

    def _mouse_release(self, event):
        """Handler for mouse button release"""
        self._event_xy = None
        if event.button == 1:
            self._button1 = False
        elif event.button == 3:
            self._button2 = False

    def _mouse_motion(self, event):
        """Handler for mouse motion"""
        if self._button1 or self._button2:
            dx = event.x - self._event_xy[0]
            dy = event.y - self._event_xy[1]
            self._event_xy = (event.x, event.y)

            if self._button1:
                if self._shift:
                    ax_LR = self._ax_LR_alt
                else:
                    ax_LR = self._ax_LR
                rot1 = Quaternion.from_v_theta(self._ax_UD,
                                               self._step_UD * dy)
                rot2 = Quaternion.from_v_theta(ax_LR,
                                               self._step_LR * dx)

                self.rotate(rot1 * rot2)

                self._draw_cube()

            if self._button2:
                factor = 1 - 0.003 * (dx + dy)
                xlim = self.get_xlim()
                ylim = self.get_ylim()
                self.set_xlim(factor * xlim[0], factor * xlim[1])
                self.set_ylim(factor * ylim[0], factor * ylim[1])

                self.figure.canvas.draw()
                
def redundant_move(move1,move2):
    # Identifies redundant moves.
    
    move_conv={'A':'T','B':'T','C':'T','D':'L','E':'L','F':'L',\
               'G':'F','H':'F','I':'F','J':'R','K':'R','L':'R',\
               'M':'B','N':'B','O':'B','P':'U','Q':'U','R':'U','Z':'Z'}
    face1=move_conv[move1]
    face2=move_conv[move2]
    
    #Two moves on same face is redundant
    #50% of opposing moves redundant - Top,Under is covered by Under,Top
    if face1==face2: redundant=1 
    elif face1=='T' and face2=='U': redundant=1    
    elif face1=='L' and face2=='R': redundant=1    
    elif face1=='F' and face2=='B': redundant=1        
    else: redundant=0
  
    return redundant
     
def cube_side_move(rcube,rcube_move):    
    
    # Moves are defined by which face, and a clockwise move 90, 180, or 270 degrees. 
    # Top90, Top180, Top270 is moves A, B and C respectively (Yellow)
    # Left90, Left180, Left270 is moves D, E and F respectively
    # Front90, Front180, Front270 is moves G, H and I respectively
    # Right90, Right180, Right270 is moves J, K and L respectively
    # Back90, Back180, Back270 is moves M, N and O respectively
    # Under90, Under180, Under270 is moves P, Q and R respectively
    
    rcube_new = list(rcube)
    if rcube_move=='A':  # 'T1':  
        rcube_new[0], rcube_new[1], rcube_new[2] =rcube[6], rcube[3], rcube[0]
        rcube_new[3], rcube_new[5]=rcube[7], rcube[1] #don't need to copy center
        rcube_new[6], rcube_new[7], rcube_new[8] =rcube[8], rcube[5], rcube[2] 
        rcube_new[9], rcube_new[10],rcube_new[11]=rcube[18],rcube[19],rcube[20]
        rcube_new[18],rcube_new[19],rcube_new[20]=rcube[27],rcube[28],rcube[29]   
        rcube_new[27],rcube_new[28],rcube_new[29]=rcube[36],rcube[37],rcube[38]  
        rcube_new[36],rcube_new[37],rcube_new[38]=rcube[9], rcube[10],rcube[11]

    elif rcube_move =='B': # 'T2':
        rcube_new[9], rcube_new[10],rcube_new[11]=rcube[27],rcube[28],rcube[29]
        rcube_new[18],rcube_new[19],rcube_new[20]=rcube[36],rcube[37],rcube[38]
        rcube_new[27],rcube_new[28],rcube_new[29]=rcube[9], rcube[10],rcube[11]  
        rcube_new[36],rcube_new[37],rcube_new[38]=rcube[18],rcube[19],rcube[20]
        rcube_new[0], rcube_new[1], rcube_new[2] =rcube[8], rcube[7], rcube[6]
        rcube_new[3],               rcube_new[5] =rcube[5],           rcube[3]        
        rcube_new[6], rcube_new[7], rcube_new[8] =rcube[2], rcube[1], rcube[0]

    elif rcube_move=='C':  # 'T3':
        rcube_new[9], rcube_new[10],rcube_new[11]=rcube[36],rcube[37],rcube[38]   
        rcube_new[18],rcube_new[19],rcube_new[20]=rcube[9], rcube[10],rcube[11]  
        rcube_new[27],rcube_new[28],rcube_new[29]=rcube[18],rcube[19],rcube[20]
        rcube_new[36],rcube_new[37],rcube_new[38]=rcube[27],rcube[28],rcube[29]
        rcube_new[0], rcube_new[1], rcube_new[2] =rcube[2], rcube[5], rcube[8]
        rcube_new[3],               rcube_new[5] =rcube[1],           rcube[7]        
        rcube_new[6], rcube_new[7], rcube_new[8] =rcube[0], rcube[3], rcube[6] 

    elif rcube_move=='D':  # 'L1':
        rcube_new[0], rcube_new[3], rcube_new[6] =rcube[44],rcube[41],rcube[38] 
        rcube_new[18],rcube_new[21],rcube_new[24]=rcube[0], rcube[3], rcube[6] 
        rcube_new[38],rcube_new[41],rcube_new[44]=rcube[51],rcube[48],rcube[45] 
        rcube_new[51],rcube_new[48],rcube_new[45]=rcube[24],rcube[21],rcube[18]
        rcube_new[9], rcube_new[10],rcube_new[11]=rcube[15],rcube[12],rcube[9]
        rcube_new[12],              rcube_new[14]=rcube[16],          rcube[10]      
        rcube_new[15],rcube_new[16],rcube_new[17]=rcube[17],rcube[14],rcube[11]

    elif rcube_move=='E': # 'L2':
        rcube_new[0], rcube_new[3], rcube_new[6] =rcube[45],rcube[48],rcube[51]
        rcube_new[18],rcube_new[21],rcube_new[24]=rcube[44],rcube[41],rcube[38]   
        rcube_new[38],rcube_new[41],rcube_new[44]=rcube[24],rcube[21],rcube[18]
        rcube_new[51],rcube_new[48],rcube_new[45]=rcube[6], rcube[3], rcube[0]
        rcube_new[9], rcube_new[10],rcube_new[11]=rcube[17],rcube[16],rcube[15]
        rcube_new[12],              rcube_new[14]=rcube[14],          rcube[12]      
        rcube_new[15],rcube_new[16],rcube_new[17]=rcube[11],rcube[10],rcube[9]

    elif rcube_move=='F':  # 'L3':
        rcube_new[0], rcube_new[3], rcube_new[6] =rcube[18],rcube[21],rcube[24] 
        rcube_new[18],rcube_new[21],rcube_new[24]=rcube[45],rcube[48],rcube[51] 
        rcube_new[38],rcube_new[41],rcube_new[44]=rcube[6], rcube[3], rcube[0] 
        rcube_new[51],rcube_new[48],rcube_new[45]=rcube[38],rcube[41],rcube[44]
        rcube_new[9], rcube_new[10],rcube_new[11]=rcube[11],rcube[14],rcube[17]
        rcube_new[12],              rcube_new[14]=rcube[10],          rcube[16]       
        rcube_new[15],rcube_new[16],rcube_new[17]=rcube[9], rcube[12],rcube[15] 
            
    elif rcube_move=='G':  # 'F1':
        rcube_new[6], rcube_new[7], rcube_new[8] =rcube[17],rcube[14],rcube[11]   
        rcube_new[11],rcube_new[14],rcube_new[17]=rcube[45],rcube[46],rcube[47]  
        rcube_new[27],rcube_new[30],rcube_new[33]=rcube[6], rcube[7], rcube[8] 
        rcube_new[45],rcube_new[46],rcube_new[47]=rcube[33],rcube[30],rcube[27] 
        rcube_new[18],rcube_new[19],rcube_new[20]=rcube[24],rcube[21],rcube[18]
        rcube_new[21],              rcube_new[23]=rcube[25] ,         rcube[19]       
        rcube_new[24],rcube_new[25],rcube_new[26]=rcube[26],rcube[23],rcube[20] 
      
    elif rcube_move=='H':   # 'F2':
        rcube_new[6], rcube_new[7], rcube_new[8] =rcube[47],rcube[46],rcube[45]
        rcube_new[11],rcube_new[14],rcube_new[17]=rcube[33],rcube[30],rcube[27] 
        rcube_new[27],rcube_new[30],rcube_new[33]=rcube[17],rcube[14],rcube[11]  
        rcube_new[45],rcube_new[46],rcube_new[47]=rcube[8], rcube[7], rcube[6] 
        rcube_new[18],rcube_new[19],rcube_new[20]=rcube[26],rcube[25],rcube[24]
        rcube_new[21],              rcube_new[23]=rcube[23],          rcube[21]        
        rcube_new[24],rcube_new[25],rcube_new[26]=rcube[20],rcube[19],rcube[18]
                 
    elif rcube_move=='I':  # 'F3':
        rcube_new[6], rcube_new[7], rcube_new[8] =rcube[27],rcube[30],rcube[33] 
        rcube_new[11],rcube_new[14],rcube_new[17]=rcube[8], rcube[7], rcube[6] 
        rcube_new[27],rcube_new[30],rcube_new[33]=rcube[47],rcube[46],rcube[45]    
        rcube_new[45],rcube_new[46],rcube_new[47]=rcube[11],rcube[14],rcube[17] 
        rcube_new[18],rcube_new[19],rcube_new[20]=rcube[20],rcube[23],rcube[26]
        rcube_new[21],              rcube_new[23]=rcube[19],          rcube[25]        
        rcube_new[24],rcube_new[25],rcube_new[26]=rcube[18],rcube[21],rcube[24]

    elif rcube_move=='J':  # 'R1':
        rcube_new[8], rcube_new[5], rcube_new[2] =rcube[26],rcube[23],rcube[20] 
        rcube_new[26],rcube_new[23],rcube_new[20]=rcube[53],rcube[50],rcube[47]  
        rcube_new[36],rcube_new[39],rcube_new[42]=rcube[8], rcube[5], rcube[2]
        rcube_new[47],rcube_new[50],rcube_new[53]=rcube[42],rcube[39],rcube[36]
        rcube_new[27],rcube_new[28],rcube_new[29]=rcube[33],rcube[30],rcube[27]
        rcube_new[30],              rcube_new[32]=rcube[34],          rcube[28]        
        rcube_new[33],rcube_new[34],rcube_new[35]=rcube[35],rcube[32],rcube[29]

    elif rcube_move=='K': # 'R2':
        rcube_new[8], rcube_new[5], rcube_new[2] =rcube[53],rcube[50],rcube[47] 
        rcube_new[26],rcube_new[23],rcube_new[20]=rcube[36],rcube[39],rcube[42] 
        rcube_new[36],rcube_new[39],rcube_new[42]=rcube[26],rcube[23],rcube[20] 
        rcube_new[47],rcube_new[50],rcube_new[53]=rcube[2], rcube[5], rcube[8]
        rcube_new[27],rcube_new[28],rcube_new[29]=rcube[35],rcube[34],rcube[33]
        rcube_new[30],              rcube_new[32]=rcube[32],rcube[30]       
        rcube_new[33],rcube_new[34],rcube_new[35]=rcube[29],rcube[28],rcube[27] 

    elif rcube_move=='L':   # 'R3':
        rcube_new[8], rcube_new[5], rcube_new[2] =rcube[36],rcube[39],rcube[42]
        rcube_new[26],rcube_new[23],rcube_new[20]=rcube[8], rcube[5], rcube[2] 
        rcube_new[36],rcube_new[39],rcube_new[42]=rcube[53],rcube[50],rcube[47] 
        rcube_new[47],rcube_new[50],rcube_new[53]=rcube[20],rcube[23],rcube[26]
        rcube_new[27],rcube_new[28],rcube_new[29]=rcube[29],rcube[32],rcube[35]
        rcube_new[30],              rcube_new[32]=rcube[28],          rcube[34]      
        rcube_new[33],rcube_new[34],rcube_new[35]=rcube[27],rcube[30],rcube[33] 

    elif rcube_move=='M':   # 'B1':
        rcube_new[2], rcube_new[1], rcube_new[0] =rcube[35],rcube[32],rcube[29]
        rcube_new[9], rcube_new[12],rcube_new[15]=rcube[2], rcube[1], rcube[0]
        rcube_new[29],rcube_new[32],rcube_new[35]=rcube[53],rcube[52],rcube[51]
        rcube_new[53],rcube_new[52],rcube_new[51]=rcube[15],rcube[12],rcube[9]
        rcube_new[36],rcube_new[37],rcube_new[38]=rcube[42],rcube[39],rcube[36]
        rcube_new[39],              rcube_new[41]=rcube[43],          rcube[37]
        rcube_new[42],rcube_new[43],rcube_new[44] =rcube[44],rcube[41],rcube[38]
        
    elif rcube_move=='N':  # 'B2':
        rcube_new[2], rcube_new[1], rcube_new[0] =rcube[51],rcube[52],rcube[53] 
        rcube_new[9], rcube_new[12],rcube_new[15]=rcube[35],rcube[32],rcube[29] 
        rcube_new[29],rcube_new[32],rcube_new[35]=rcube[15],rcube[12],rcube[9]
        rcube_new[53],rcube_new[52],rcube_new[51]=rcube[0], rcube[1], rcube[2]
        rcube_new[36],rcube_new[37],rcube_new[38]=rcube[44],rcube[43],rcube[42]
        rcube_new[39],              rcube_new[41]=rcube[41],          rcube[39]       
        rcube_new[42],rcube_new[43],rcube_new[44]=rcube[38],rcube[37],rcube[36] 

    elif rcube_move=='O':  # 'B3':
        rcube_new[2], rcube_new[1], rcube_new[0] =rcube[9], rcube[12],rcube[15] 
        rcube_new[9], rcube_new[12],rcube_new[15]=rcube[51],rcube[52],rcube[53]
        rcube_new[29],rcube_new[32],rcube_new[35]=rcube[0], rcube[1], rcube[2]
        rcube_new[53],rcube_new[52],rcube_new[51]=rcube[29],rcube[32],rcube[35]
        rcube_new[36],rcube_new[37],rcube_new[38]=rcube[38],rcube[41],rcube[44]
        rcube_new[39],              rcube_new[41]=rcube[37],          rcube[43]       
        rcube_new[42],rcube_new[43],rcube_new[44]=rcube[36],rcube[39],rcube[42]  
        
    elif rcube_move=='P': # 'U1':
        rcube_new[15],rcube_new[16],rcube_new[17]=rcube[42],rcube[43],rcube[44] 
        rcube_new[24],rcube_new[25],rcube_new[26]=rcube[15],rcube[16],rcube[17] 
        rcube_new[33],rcube_new[34],rcube_new[35]=rcube[24],rcube[25],rcube[26]              
        rcube_new[42],rcube_new[43],rcube_new[44]=rcube[33],rcube[34],rcube[35]
        rcube_new[45],rcube_new[46],rcube_new[47]=rcube[51],rcube[48],rcube[45]
        rcube_new[48],              rcube_new[50]=rcube[52],          rcube[46]       
        rcube_new[51],rcube_new[52],rcube_new[53]=rcube[53],rcube[50],rcube[47] 

    elif rcube_move=='Q':   # 'U2':
        rcube_new[15],rcube_new[16],rcube_new[17]=rcube[33],rcube[34],rcube[35] 
        rcube_new[24],rcube_new[25],rcube_new[26]=rcube[42],rcube[43],rcube[44]  
        rcube_new[33],rcube_new[34],rcube_new[35]=rcube[15],rcube[16],rcube[17]  
        rcube_new[42],rcube_new[43],rcube_new[44]=rcube[24],rcube[25],rcube[26] 
        rcube_new[45],rcube_new[46],rcube_new[47]=rcube[53],rcube[52],rcube[51]
        rcube_new[48],              rcube_new[50]=rcube[50],          rcube[48]     
        rcube_new[51],rcube_new[52],rcube_new[53]=rcube[47],rcube[46],rcube[45]
 
    elif rcube_move=='R':  # 'U3':
        rcube_new[15],rcube_new[16],rcube_new[17]=rcube[24],rcube[25],rcube[26] 
        rcube_new[24],rcube_new[25],rcube_new[26]=rcube[33],rcube[34],rcube[35]  
        rcube_new[33],rcube_new[34],rcube_new[35]=rcube[42],rcube[43],rcube[44]
        rcube_new[42],rcube_new[43],rcube_new[44]=rcube[15],rcube[16],rcube[17]
        rcube_new[45],rcube_new[46],rcube_new[47]=rcube[47],rcube[50],rcube[53]
        rcube_new[48],              rcube_new[50]=rcube[46],          rcube[52]       
        rcube_new[51],rcube_new[52],rcube_new[53]=rcube[45],rcube[48],rcube[51]

    return ''.join(rcube_new) 

def score_rcube(rcube,verbose=0):
    # To get to 100 score, edges need to be in solvable locations
    # Once at score 100, edges and corners checked for correct location
    # center cubes do not need to be checked
   
    bad_edges=0
    layer2_pieces=0
   
    edge1 =rcube[1] +rcube[37]  #tb
    edge2 =rcube[3] +rcube[10]  #tl
    edge3 =rcube[5] +rcube[28]  #tr
    edge4 =rcube[7] +rcube[19]  #tf
    edge5 =rcube[12]+rcube[41]  #lb
    edge6 =rcube[14]+rcube[21]  #lf
    edge7 =rcube[16]+rcube[48]  #lu
    edge8 =rcube[23]+rcube[30]  #fr
    edge9 =rcube[25]+rcube[46]  #fu
    edge10=rcube[32]+rcube[39]  #rb
    edge11=rcube[34]+rcube[50]  #ru
    edge12=rcube[43]+rcube[52]  #bu

    #Following lists generated by testing what edges are possible within 6 moves of solved
    if edge7  not in ['lu','ru']: bad_edges+=1
    if edge11 not in ['lu','ru']: bad_edges+=1  
    if verbose == 1: print "7 and 11 are (2)",bad_edges
    
    if edge1  not in ['tb','tl','tr','tf','fr','bl','fl','br']: bad_edges+=1 
    if edge2  not in ['tb','tl','tr','tf','fr','bl','fl','br']: bad_edges+=1
    if edge3  not in ['tb','tl','tr','tf','fr','bl','fl','br']: bad_edges+=1
    if edge4  not in ['tb','tl','tr','tf','fr','bl','fl','br']: bad_edges+=1
    if verbose == 1: print "Add 1,2,3,4 (6)",bad_edges
    
    if edge8  not in ['tb','tl','tr','tf','fr','bl','fl','br']: bad_edges+=1
    if edge5  not in ['lb','lf','rb','bt','lt','rt','ft','rf']: bad_edges+=1
    if verbose == 1: print "Add 8,5 (8)",bad_edges   
    
    if edge6  not in ['lb','lf','rb','bt','lt','rt','ft','rf']: bad_edges+=1
    if edge10 not in ['lb','lf','rb','bt','lt','rt','ft','rf']: bad_edges+=1
    if verbose ==1: print "Add 6,10 (10)",bad_edges
    
    if edge9 <> 'fu': bad_edges += 2  #needs to be investigated for score. 3 best 
    if edge12 <> 'bu': bad_edges += 2
    if verbose ==1: print "Add 9,12 (30)",bad_edges    
    
      
    if bad_edges > 0:
        solve_score = 100-bad_edges
    else:
        corner1=rcube[0] +rcube[38]+rcube[9]
        corner2=rcube[2] +rcube[36]+rcube[29]
        corner3=rcube[6] +rcube[10]+rcube[18]
        corner4=rcube[8] +rcube[20]+rcube[27]
        corner5=rcube[15]+rcube[44]+rcube[51] 
        corner6=rcube[17]+rcube[45]+rcube[24]
        corner7=rcube[26]+rcube[47]+rcube[33] 
        corner8=rcube[35]+rcube[53]+rcube[42] 
 
        if edge5 =='lb': layer2_pieces+=1       
        if edge6 =='lf': layer2_pieces+=1   
        if edge7 =='lu': layer2_pieces+=1                        
        if edge8 =='fr': layer2_pieces+=1
        if edge10=='rb': layer2_pieces+=1   
        if edge11=='ru': layer2_pieces+=1
        if corner5=='lbu': layer2_pieces+=1     
        if corner6=='luf': layer2_pieces+=1
        if corner7=='fur': layer2_pieces+=1
        if corner8=='rub': layer2_pieces+=1   
        
        #Next loop Needed as solving layer 2 and 3 together is too long
        layer3_pieces = 0
        if layer2_pieces == 10:    
            if corner1 == 'tbl': layer3_pieces+=1  #
            if corner2 == 'tbr': layer3_pieces+=1  
            if corner3 == 'tlf': layer3_pieces+=1  
            if corner4 == 'tfr': layer3_pieces+=1   
            if edge1 == 'tb': layer3_pieces+=1  
            if edge2 == 'tl': layer3_pieces+=1  
            if edge3 == 'tr': layer3_pieces+=1
            if edge4 == 'tf': layer3_pieces+=1  
        
        solve_score=100+layer2_pieces+layer3_pieces
    return solve_score

    
def fetch_vertex(rcube,move_sequence,level_count,num_levels,available_cube_moves):

    next_cubes = []
    next_moves = []
    
    last_move = move_sequence[-1]
    if (level_count != num_levels):
        for next_move in available_cube_moves:
            if not redundant_move(last_move,next_move):
                next_cubes.append(cube_side_move(rcube,next_move)) #create a list of cubes
                move_history = move_sequence+next_move
                next_moves.append(move_history)

    return next_cubes,next_moves

def make_moves (rcube_start, moves):
    #Executes a series of moves. 
    new_cube = rcube_start
    if moves[0]=='Z': moves=moves[1:] #remove the start char
    for next_move in moves:
        new_cube = cube_side_move(new_cube,next_move)
    return new_cube
                   
def compress_cube(rcube):
    
    #returns a base 6 integer unique to the cube
    cube_char_to_int ={'t':'0','l':'1','f':'2','r':'3','b':'4','u':'5'}

    rcube_int_string = ''
    counter = 0
    for i in range(54):
        if i not in [4,13,22,31,40,49]: #centers not required
            rcube_int_string = rcube_int_string + cube_char_to_int[rcube[i]]
            counter += 1
    
    return int(rcube_int_string,base=6)  
         
def create_cube_dict(current_location, rcube, depth, tree_max_depth, cubes_dict):
    
    rcube_base6 = compress_cube(rcube)
    
    #Add to dictionary if no entry, or current location is shorter
    if rcube_base6 in cubes_dict:
        if len(cubes_dict[rcube_base6])>len(current_location)-1: 
            update_dict=True
        else: 
            update_dict = False 
    else:
        update_dict=True
    
    if update_dict==True: 
        reverse_move_dict={'A':'C','B':'B','C':'A','D':'F','E':'E','F':'D',\
                           'G':'I','H':'H','I':'G','J':'L','K':'K','L':'J',\
                           'M':'O','N':'N','O':'M','P':'R','Q':'Q','R':'P'}
        moves_to_solved=current_location[:0:-1]
        reverse_moves = ''
        for mychar in moves_to_solved: 
            reverse_moves=reverse_moves+reverse_move_dict[mychar]
        cubes_dict.update({rcube_base6:reverse_moves})

    if depth < tree_max_depth:
        #if depth <5:
        #    available_cube_moves = ['A','B','C','D','E','F','G','H','I','J',\
        #                        'K','L','M','N','O','P','Q','R']
        #else:               # ['A','B','C','D','E','F','J','K','I']  
        available_cube_moves = ['B','E','K','A','D','L','C','F','J']     
                 
        for cmove in available_cube_moves:
            if depth == 0: print "Creating dict for 9 letters, at letter",\
                           cmove,". Length of dictionary is ",len(cubes_dict)
            if (not redundant_move(current_location[-1],cmove)):
                next_location = current_location+cmove
                recursion_rcube = rcube 
                create_cube_dict(next_location, \
                                 cube_side_move(recursion_rcube,cmove),\
                                 depth+1,tree_max_depth,cubes_dict)
    return 0 

def worker_function(nodes_to_visit, cubes_queue, moves_queue, rcube_start,\
                    num_levels,visited,level_count,cubes_dict,\
                    record_cube,record_cube_loc,start_record_score,phase): 
    
    # Top90, Top180, Top270 is moves A, B and C respectively (Yellow)
    # Left90, Left180, Left270 is moves D, E and F respectively
    # Front90, Front180, Front270 is moves G, H and I respectively
    # Right90, Right180, Right270 is moves J, K and L respectively
    # Back90, Back180, Back270 is moves M, N and O respectively
    # Under90, Under180, Under270 is moves P, Q and R respectively MPMRJN   GPGRDH MGPRJDNH

    
    improvement_found = False
    success = False

    while len(nodes_to_visit) != 0:
        #print 'To visit is',len(nodes_to_visit),'\r',
        node = nodes_to_visit.popleft()
        cube = make_moves(rcube_start,node)
        best_score = score_rcube(record_cube)
   
        #If phase is 1, cube will not be in dictionary so only check for success.  
        if phase == 1:
            if cube=='tttttttttlllllllllfffffffffrrrrrrrrrbbbbbbbbbuuuuuuuuu':
                success=True
                record_cube_loc=node
                print "Success! Cube is",cube,"moves are",record_cube_loc
                return cube,record_cube_loc, improvement_found, success
            
        else:
            cube_sml = compress_cube(cube)
            if cube_sml in cubes_dict:
                success = True
                record_cube_loc = node+cubes_dict[cube_sml]
                #print "Dictionary find!!!!!. Cube is ",cube,\
                #"Dict says moves",cubes_dict[cube_sml],"Moves is",record_cube_loc
                
                return cube,record_cube_loc, improvement_found, success
            
        cube_score= score_rcube(cube)  

        if cube_score > best_score: 
            record_cube=cube
            record_cube_loc = node
            print "Better score found ",cube_score, "at location ",record_cube_loc
        
        if level_count > (3+phase*2):
            if best_score > start_record_score:
                print "Score is ",best_score, "Cube is ",record_cube
                improvement_found = True
                return record_cube,record_cube_loc, improvement_found, success

        if (node not in visited): 
            if phase == 1: available_cube_moves =  ['M','G','P','R','J','D',\
                                                   'B','E','H','K','A','N','Q',\
                                                   'C','F','I','L','O','R'] 

            elif phase == 2: available_cube_moves = ['B','K','F','A','C','D','J','L','E']         
            next_cubes,next_moves = fetch_vertex(cube,node, level_count,\
                                                 num_levels,available_cube_moves)
            if next_cubes != None: 
                cubes_queue.extend(next_cubes)
                moves_queue.extend(next_moves)
                        
    return record_cube,record_cube_loc, improvement_found, success

def solve_rubiks_cube(rcube):

    #how deep to search for phase 1
    num_levels = 8
    
    #how deep to search back from solved. 10 if you have a 8GB machine and no presaved file
    preload_depth = 10 

    solved_cube = 'tttttttttlllllllllfffffffffrrrrrrrrrbbbbbbbbbuuuuuuuuu'
    cubes_dict = {}
    record_cube=rcube
    record_cube_loc = 'Z'
    true_location = 'Z'
    start_record_score = score_rcube(record_cube)
        
    print "Starting cube is",rcube," score is",start_record_score," at ",time.asctime()    
    cubefilename = 'cube'+str(preload_depth)+'.p'
    if  not os.path.exists(cubefilename):
        print "No solve dictionary file, creating it for depth",preload_depth, "Takes approx 10 mins, starting at", time.asctime()
        create_cube_dict('Z',  solved_cube, 0, preload_depth,cubes_dict)   
        print  "Created list, starting save at ",time.asctime()
        cubefile = open(cubefilename, 'wb')
        cPickle.dump(cubes_dict,cubefile,protocol = cPickle.HIGHEST_PROTOCOL)
        print "Save is complete"
    else: 
        print "Reading rubiks dictionary "
        cubefile = open(cubefilename, 'rb')
        cubes_dict = cPickle.load(cubefile)
        print "Loaded dictionary",time.asctime()
    cubefile.close()   
    improvement_found = True
    success = False
    phase = 1
    while (improvement_found == True and success == False):
        print ""
        print "Starting at depth 1....."
        improvement_found = False
        bfs_queue = collections.deque(['Z'])     # Z denotes start
        level_count = 0
        #visited = []
        visited = set()
        
        while (len(bfs_queue) != 0 and level_count < num_levels) and improvement_found == False:
            level_count += 1 
            cubes_queue = collections.deque()
            moves_queue = collections.deque()
            tmp = list(bfs_queue)[0:]
            print "Depth is",level_count,"at:",time.asctime()
            
            record_cube,record_cube_loc, improvement_found,success = \
            worker_function(bfs_queue, cubes_queue, moves_queue, rcube, num_levels, visited,\
                            level_count,cubes_dict,record_cube,record_cube_loc,start_record_score,phase)
            
            if (not success):

                if (improvement_found):
                    rcube = record_cube
                    true_location = true_location+record_cube_loc[1:]
                    start_record_score = score_rcube(record_cube)
                    if start_record_score>=100 and phase == 1: 
                        print " "
                        print "Starting phase 2......"
                        phase = 2
                        num_levels = 11
                    
            else:
                improvement_found = True
                true_location = true_location+record_cube_loc[1:]
                print "Success!!!!!! Number of moves is ",len(true_location)-1, "location is ",true_location,"at ",time.asctime() 
        
            #visited.extend(tmp)
            visited.update(tmp)
            
            bfs_queue = moves_queue
            
    if (not success):
        return 0
    else: 
        return condense_moves(true_location[1:])
    
    
def condense_moves(moves):
    
    """Looks for moves that can be shortened """
    """Occurs due to appending move sequences"""
    
    # Moves are defined by which face, and a clockwise move 90, 180, or 270 degrees. 
    # Top90, Top180, Top270 is moves A, B and C respectively (Yellow)
    # Left90, Left180, Left270 is moves D, E and F respectively
    # Front90, Front180, Front270 is moves G, H and I respectively
    # Right90, Right180, Right270 is moves J, K and L respectively
    # Back90, Back180, Back270 is moves M, N and O respectively
    # Under90, Under180, Under270 is moves P, Q and R respectively
    
    
    replace_moves={'AA':'B','BB':'','CC':'B','DD':'E','EE':'','FF':'E','AB':'C','AC':'','BA':'C','BC':'A','CA':'','CB':'A',\
                   'GG':'H','HH':'','II':'H','JJ':'K','KK':'','LL':'K',\
                   'MM':'N','NN':'','OO':'N','PP':'Q','QQ':'','RR':'Q',\
                   'APA':'BP','AQA':'BQ','ARA':'BR','BPB':'P','BQB':'Q',\
                   'BRB':'R','CPC':'BP','CQC':'BQ','CRC':'BR','DJD':'EJ',\
                   'DKD':'EK','DLD':'EL','EJE':'J','EKE':'K','ELE':'L',\
                   'FJF':'EJ','FKF':'EK','FLF':'EL','JDJ':'KD','JEJ':'KE',\
                   'JFJ':'KF','KDK':'D','KEK':'E','KFK':'F','LDL':'KD',\
                   'LEL':'KE','LFL':'KF'}              
    
    replacements_found = True
    temp_moves = moves
    new_moves = moves
    
    while replacements_found==True:
        for key in replace_moves.iterkeys():
            temp_moves=temp_moves.replace(key,replace_moves[key])
        
        if len(temp_moves)<len(new_moves): new_moves=temp_moves
        else: replacements_found=False
        
    return new_moves
        
def profile_solve():
    
    """ Used to test speed of program """
    
    start_time = time.time()
    cubes_list=[]
    
    cubes_list.append('lurrtflutftblltfububbffrrbblrfurrutfulubbblfrttrfultlt')
    cubes_list.append('luuftbbtfbuullbflrllttfrbtflurfruubtfrtlbfbfrurlburttr')    
    cubes_list.append('ftbftlftlrrlllbfruuftufrrbfbbturltrurlufbubftbtrtublul')
    cubes_list.append('brfttrfturbrllbtlltflufutlrbflfrrbbuuttubtfubfbufurllr')
    cubes_list.append('lltutlruuurbtlbttftfblfbuffrffrrrrblrubfbrbulltulutfbt')
    cubes_list.append('rrlftubbutrrtlutrrutlfftflfblffruurrttbbblubbtflbuullf')
    cubes_list.append('uufbtrblufutblfltbrbfuftuuultrfrlbtltrrfbrbffrlllubtrt')
    cubes_list.append('tbfftturtbubrlfblurfflflfburftbrllrlltrubuutlruftubtrb')
    cubes_list.append('rtffttbtburtbllrllrllffbtfltfllrruububftbrubtfubuurfur')
 
    for cube_sample in cubes_list:
        solve_rubiks_cube(cube_sample)
        
    print "Time taken is",time.time()-start_time
       

if __name__ == '__main__':
    
    #cube is NxN
    try:
        N = int(sys.argv[1])
    except:
        N = 3
        
    start_time=time.time()
    mycolors=[]
    
    for i in range(6):
        colors_i = i + np.zeros(N*N, dtype=int)
        mycolors.append(colors_i)

    plastic_color=None
    face_colors=None
        
    c = Cube(N,plastic_color,face_colors,mycolors)
    c.draw_interactive()
    plt.show()
