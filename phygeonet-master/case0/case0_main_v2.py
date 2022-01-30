import Ofpp
import torch

from model import USCNN
from pyMesh import hcubeMesh

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
h = 0.01
OFBCCoord = Ofpp.parse_boundary_field('TemplateCase/30/C')
OFLOWC = OFBCCoord[b'low'][b'value']
OFUPC = OFBCCoord[b'up'][b'value']
OFLEFTC = OFBCCoord[b'left'][b'value']
OFRIGHTC = OFBCCoord[b'right'][b'value']
leftX = OFLEFTC[:, 0]
leftY = OFLEFTC[:, 1]
lowX = OFLOWC[:, 0]
lowY = OFLOWC[:, 1]
rightX = OFRIGHTC[:, 0]
rightY = OFRIGHTC[:, 1]
upX = OFUPC[:, 0]
upY = OFUPC[:, 1]
ny = len(leftX)
nx = len(lowX)
myMesh = hcubeMesh(leftX, leftY, rightX, rightY, lowX, lowY,
                   upX, upY, h, True, True, tolMesh=1e-10, tolJoint=1)

batchSize = 1
NvarInput = 2
NvarOutput = 1
nEpochs = 1500
lr = 0.001
Ns = 1
nu = 0.01
model = USCNN(h, nx, ny, NvarInput, NvarOutput).to(device)
