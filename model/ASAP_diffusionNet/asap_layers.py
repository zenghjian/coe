# 3p
import torch
import torch.nn as nn
# project
from .asap_utils import to_basis, from_basis, ensure_complex, cmatvecmul_stacked


class LaplacianBlock(nn.Module):
    """
    Applies Laplacian powers/diffusion in the spectral domain like
        f_out = lambda_i ^ k * e ^ (lambda_i t) f_in
    with learned per-channel parameters k and t.

    Inputs:
      - values: (K,C) in the spectral domain
      - evals: (K) eigenvalues
    Outputs:
      - (K,C) transformed values in the spectral domain
    """

    def __init__(self, C_inout, with_power=True, max_time=False):
        super(LaplacianBlock, self).__init__()
        self.C_inout = C_inout
        self.with_power = with_power
        self.max_time = max_time

        self.laplacian_power = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)

        if self.with_power:
            nn.init.constant_(self.laplacian_power, 0.0)
        nn.init.constant_(self.diffusion_time, 0.0001)

    def forward(self, x, evals):

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout
                )
            )

        if self.max_time:
            diffusion_time = self.max_time * torch.sigmoid(self.diffusion_time)
            # diffusion_time = self.diffusion_time.clamp(min=-self.max_time, max=self.max_time)
        else:
            diffusion_time = self.diffusion_time

        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * torch.abs(diffusion_time).unsqueeze(0))

        if self.with_power:
            lambda_coefs = torch.pow(evals.unsqueeze(-1), (2.0 * torch.sigmoid(self.laplacian_power) - 1.0).unsqueeze(0))
        else:
            lambda_coefs = torch.ones_like(self.laplacian_power)

        if x.is_complex():
            y = ensure_complex(lambda_coefs * diffusion_coefs) * x
        else:
            y = lambda_coefs * diffusion_coefs * x

        return y


class PairwiseDot(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.

    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots
    """

    def __init__(self, C_inout, linear_complex=True):
        super(PairwiseDot, self).__init__()

        self.C_inout = C_inout
        self.linear_complex = linear_complex

        if self.linear_complex:
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

    def forward(self, vectors):

        vectorsA = vectors  # (V,C)

        if self.linear_complex:
            vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vectorsBreal = self.A(vectors[..., 0])
            vectorsBimag = self.A(vectors[..., 1])

        dots = vectorsA[..., 0] * vectorsBreal + vectorsA[..., 1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    def __init__(
        self,
        layer_sizes,
        dropout=False,
        activation=nn.ReLU,
        batch_norm=False,
        name="miniMLP",
    ):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = i + 2 == len(layer_sizes)

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i), nn.Dropout(p=0.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i + 1],),
            )

            if batch_norm and not is_last:
                self.add_module(
                    name + "_mlp_batch_norm_{:03d}".format(i),
                    BatchNormLastDim(layer_sizes[i + 1]),
                )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(name + "_mlp_act_{:03d}".format(i), activation())


class BatchNormLastDim(nn.Module):
    def __init__(self, s):
        super(BatchNormLastDim, self).__init__()
        self.s = s
        self.bn = nn.BatchNorm1d(s)

    def forward(self, x):
        init_dim = x.shape
        if init_dim[-1] != self.s:
            raise ValueError(
                "batch norm last dim does not have right shape. should be {}, but is {}".format(
                    self.s, init_dim[-1]
                )
            )

        x_flat = x.view((-1, self.s))
        bn_flat = self.bn(x_flat)
        return bn_flat.view(*init_dim)


class TSNBlock_Scalar(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(
        self,
        C0_inout,
        C0_hidden,
        dropout=False,
        pairwise_dot=True,
        with_power=False,
        max_time=False,
        dot_linear_complex=True,
        grad_method="spectral_pointwise",
    ):
        super(TSNBlock_Scalar, self).__init__()

        # Specified dimensions
        self.C0_inout = C0_inout
        self.C0_hidden = C0_hidden

        self.dropout = dropout
        self.pairwise_dot = pairwise_dot
        self.with_power = with_power
        self.max_time = max_time
        self.dot_linear_complex = dot_linear_complex
        self.grad_method = grad_method  # one of ['pointwise', 'spectral_pointwise', 'spectral_spectral']

        # Laplacian block
        self.spec0 = LaplacianBlock(self.C0_inout, self.with_power, self.max_time)

        self.C0_mlp = 2 * self.C0_inout

        if self.pairwise_dot:
            if self.grad_method == "pointwise":
                self.pairwise_dot = PairwiseDot(
                    self.C0_inout, linear_complex=self.dot_linear_complex
                )
                self.C0_mlp += self.C0_inout
            elif self.grad_method == "spectral_pointwise":
                self.pairwise_dot = PairwiseDot(
                    self.C0_inout, linear_complex=self.dot_linear_complex
                )
                self.C0_mlp += self.C0_inout
            elif self.grad_method == "spectral_spectral":
                self.pairwise_dot = PairwiseDot(
                    self.C0_inout, linear_complex=self.dot_linear_complex
                )
                self.C0_mlp += self.C0_inout

        # MLPs
        self.mlp0 = MiniMLP([self.C0_mlp] + self.C0_hidden + [self.C0_inout], dropout=self.dropout)

    def forward(self, x0, mass, evals, evecs, gradX, gradY, grad_from_spectral=None):
        
        batch_size = x0.shape[0]
        if x0.shape[-1] != self.C0_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x0.shape, self.C0_inout
                )
            )

        # Transform to spectral
        x0_spec = to_basis(x0, evecs, mass)  # (K, C0_in)r
        
        # Laplacian block
        x0_spec = self.spec0(x0_spec, evals)

        # Transform back to per-vertex
        x0_lap = from_basis(x0_spec, evecs) # x_diffuse (V, C0_in)



        x0_grads = []
        for b in range(batch_size): 
            # gradient after diffusion
            x0_gradX = torch.mm(gradX[b,...], x0_lap[b,...])
            x0_gradY = torch.mm(gradY[b,...], x0_lap[b,...])
            x0_grads.append(torch.stack((x0_gradX, x0_gradY), dim=-1))

        x0_grad = torch.stack(x0_grads, dim=0)



        x0_comb = torch.cat(
            (x0, x0_lap), dim=-1
        )  # (V, C0_speccomb + C0_in)r = (V, C0_mlp)r

        if self.pairwise_dot:
            # If using the pairwise dot block, add it to the scalar values as well

            if self.grad_method == "spectral_pointwise":
                # x0_grad = cmatvecmul_stacked(grad_from_spectral, x0_spec)
                pass
            elif self.grad_method == "pointwise":
                pass
            elif self.grad_method == "spectral_spectral":
                pass

            x0_gradprods = self.pairwise_dot(x0_grad)
            x0_comb = torch.cat((x0_comb, x0_gradprods), dim=-1)

        # Apply the mlp
        x0_out = self.mlp0(x0_comb)

        # Skip connection
        x0_out = x0_out + x0

        return x0_out
    

class ASAP_DiffusionNet(nn.Module):

    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, outputs_at='vertices', mlp_hidden_dims=None,
                       with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral', pairwise_dot=True, dot_linear_complex=True, dropout=False):   
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(ASAP_DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block =  TSNBlock_Scalar(C0_inout=C_width,
                                C0_hidden=mlp_hidden_dims,
                                pairwise_dot=pairwise_dot,
                                dot_linear_complex=dot_linear_complex,
                                dropout=dropout
                                )

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    def forward(self, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        # Apply each of the blocks
        for b in self.blocks:
            # x = b(x, mass, L, evals, evecs, gradX, gradY)
            x = b(x, mass, evals, evecs, gradX, gradY)
        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == 'vertices': 
            x_out = x
        
        elif self.outputs_at == 'edges': 
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        
        elif self.outputs_at == 'faces': 
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)
        
        elif self.outputs_at == 'global_mean': 
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant. 
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out    